"""extract.py — posts.jsonl → facts.jsonl（LLM 抽取 + 校验 + 别名映射）。

核心 extract_post(post, llm_fn) 可注入 llm_fn(system, user)->str，生产走真实
client、测试注入 canned 响应。LLM 只吐 raw 字符串，KB id 映射由代码用 catalog
别名表做（不让 LLM 输出 KB id，降低幻觉面）。

CLI:  python -m kb_mining.extract [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from kb_mining import catalog

DEFAULT_IN = Path("kb_mining/data/posts.jsonl")
DEFAULT_OUT = Path("kb_mining/data/facts.jsonl")
DEFAULT_REJECTS = Path("kb_mining/data/extract_rejects.jsonl")

# 截断：正文超 12k 字符时保留前 9k + 后 3k（配置在开头、分数表在结尾）
TRUNC_LIMIT = 12_000
TRUNC_HEAD = 9_000
TRUNC_TAIL = 3_000
MAX_MEMBERS = 12
VALID_KINDS = {"single", "ensemble", "unclear"}

SYSTEM_PROMPT = """\
You are extracting structured facts from a Kaggle competition solution write-up.
Return pure JSON only — no markdown fences, no commentary.

Output schema:
{
  "kind": "single" | "ensemble" | "unclear",
  "members": [{"raw_model": "<model name exactly as written>", "image_size": <int|null>}],
  "loss_raw": "<loss name exactly as written>" | null,
  "best_single_model_raw": "<model name>" | null,
  "best_single_score": <float|null>,
  "used_pseudo_labeling": true | false,
  "used_tta": true | false,
  "citations": ["<verbatim sentence copied from the post>"]
}

Rules:
1. members: every distinct model architecture in the FINAL submission only —
   ignore abandoned experiments. Copy names exactly as written
   (e.g. "tf_efficientnet_b4_ns"); do NOT normalize or expand them.
2. kind: "single" if the final submission is one model; "ensemble" if it
   averages/stacks several; "unclear" if you cannot tell.
3. loss_raw: the training loss of the main model(s); null if never stated.
4. best_single_model_raw / best_single_score: only if the post explicitly
   reports a best single-model score (e.g. "our best single model scored
   0.899"); otherwise null.
5. citations: 1-3 quotes copied character-for-character from the post that
   mention the models or the loss. They are used for automatic verification;
   paraphrased quotes will cause the whole extraction to be rejected.
6. If the post is not actually a solution write-up (e.g. a question or a
   congratulations thread), return {"kind": "unclear", "members": []}.
"""


# ── 文本处理 ────────────────────────────────────────────────────────────────
def truncate_text(text: str) -> str:
    """超长正文保留头 9k + 尾 3k。"""
    if len(text) <= TRUNC_LIMIT:
        return text
    return text[:TRUNC_HEAD] + "\n...[TRUNCATED]...\n" + text[-TRUNC_TAIL:]


def _norm_ws(s: str) -> str:
    """空白归一（用于引用子串匹配）。"""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _strip_fences(raw: str) -> str:
    """去掉 LLM 偶发的 ```json ... ``` 围栏。"""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    return s


# ── 校验 ────────────────────────────────────────────────────────────────────
def validate_extraction(data: dict, text: str) -> str | None:
    """校验 LLM 输出。通过返回 None，否则返回拒绝原因字符串。"""
    if not isinstance(data, dict):
        return "not_a_dict"
    if data.get("kind") not in VALID_KINDS:
        return "bad_kind"
    members = data.get("members")
    if not isinstance(members, list) or not members:
        return "empty_members"      # 含非方案帖（unclear + []）
    if len(members) > MAX_MEMBERS:
        return "too_many_members"
    if not all(isinstance(m, dict) and m.get("raw_model") for m in members):
        return "malformed_member"
    # 引用校验：至少一条 citation 在正文中能找到（空白归一后子串匹配）
    cites = data.get("citations") or []
    norm_text = _norm_ws(text)
    if not any(c and _norm_ws(c) in norm_text for c in cites):
        return "citation_not_found"
    return None


# ── 别名映射后处理 ───────────────────────────────────────────────────────────
def postprocess(data: dict) -> dict:
    """把 LLM 的 raw 字段映射为 KB id：家族去重、image_size 众数、loss/best 映射。"""
    members = data.get("members", [])
    # 家族去重 + 每家族 image_size 众数
    fam_sizes: dict[str, list[int]] = {}
    families_order: list[str] = []
    for m in members:
        fam = catalog.map_model(m.get("raw_model"))
        if fam not in families_order:
            families_order.append(fam)
        sz = m.get("image_size")
        if isinstance(sz, (int, float)) and sz:
            fam_sizes.setdefault(fam, []).append(int(sz))
    family_image_size = {
        fam: Counter(sizes).most_common(1)[0][0]
        for fam, sizes in fam_sizes.items()
    }
    loss_raw = data.get("loss_raw")
    best_raw = data.get("best_single_model_raw")
    return {
        "kind": data["kind"],
        "families": families_order,
        "family_image_size": family_image_size,
        "loss_raw": loss_raw,
        "loss_kb": catalog.map_loss(loss_raw),
        "loss_is_metric_learning": catalog.is_metric_learning_loss(loss_raw),
        "best_single_model_raw": best_raw,
        "best_single_family": catalog.map_model(best_raw) if best_raw else None,
        "best_single_score": data.get("best_single_score"),
        "used_pseudo_labeling": bool(data.get("used_pseudo_labeling")),
        "used_tta": bool(data.get("used_tta")),
        "members_raw": members,
        "citations": data.get("citations") or [],
    }


# ── 单篇抽取 ────────────────────────────────────────────────────────────────
def classify_post(post: dict, llm_fn: Callable[[str, str], str]) -> dict:
    """抽取单篇 → {"ok": bool, "fact": dict|None, "reason": str|None}。"""
    text = post.get("text", "")
    user = truncate_text(text)
    raw = llm_fn(SYSTEM_PROMPT, user)
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "fact": None, "reason": "json_parse"}
    reason = validate_extraction(data, text)
    if reason:
        return {"ok": False, "fact": None, "reason": reason}
    parsed = postprocess(data)
    # fact = post 行 + 解析结果（raw 全保留）
    fact = {**post, **parsed}
    fact.pop("text", None)   # 正文不入 facts（体积大，需要时回 posts.jsonl 查）
    return {"ok": True, "fact": fact, "reason": None}


def extract_post(post: dict, llm_fn: Callable[[str, str], str]) -> dict | None:
    """抽取单篇 → fact dict，拒绝则 None（计划规定的公开签名）。"""
    return classify_post(post, llm_fn)["fact"]


def remap_fact(fact: dict) -> dict:
    """用当前别名表对一条已有 fact 重新映射（免 LLM）。raw 字段已保留在 fact 中。"""
    data = {
        "kind": fact["kind"],
        "members": fact.get("members_raw", []),
        "loss_raw": fact.get("loss_raw"),
        "best_single_model_raw": fact.get("best_single_model_raw"),
        "best_single_score": fact.get("best_single_score"),
        "used_pseudo_labeling": fact.get("used_pseudo_labeling"),
        "used_tta": fact.get("used_tta"),
        "citations": fact.get("citations"),
    }
    parsed = postprocess(data)
    # 保留 post 级字段（competition/topic_id/rank/...），覆盖解析字段
    post_fields = {k: v for k, v in fact.items() if k not in parsed}
    return {**post_fields, **parsed}


def remap_facts(path: Path = DEFAULT_OUT) -> int:
    """就地重映射 facts.jsonl（别名表更新后调用，无需重跑 LLM）。返回条数。"""
    facts = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    remapped = [remap_fact(f) for f in facts]
    with path.open("w", encoding="utf-8") as fh:
        for f in remapped:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"[extract] remapped {len(remapped)} facts in {path}")
    return len(remapped)


# ── 生产 LLM 客户端 ─────────────────────────────────────────────────────────
def make_real_llm_fn() -> Callable[[str, str], str]:
    """复用 Module 1 的 provider/client，返回 llm_fn(system, user)->str。"""
    from features_extraction_api import _client_for_provider, _provider

    client, model = _client_for_provider(_provider())

    def llm_fn(system: str, user: str) -> str:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            import cost_meter
            cost_meter.record_llm_call(cost_meter.tokens_from_response(completion))
        except Exception:
            pass
        return completion.choices[0].message.content

    return llm_fn


# ── 编排 ────────────────────────────────────────────────────────────────────
def run_extract(
    in_path: Path = DEFAULT_IN,
    out_path: Path = DEFAULT_OUT,
    rejects_path: Path = DEFAULT_REJECTS,
    llm_fn: Callable[[str, str], str] | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """posts.jsonl → facts.jsonl + extract_rejects.jsonl。返回 (facts, rejects)。"""
    llm_fn = llm_fn or make_real_llm_fn()
    posts = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if limit:
        posts = posts[:limit]

    facts: list[dict] = []
    rejects: list[dict] = []
    for post in posts:
        try:
            r = classify_post(post, llm_fn)
        except Exception as e:  # LLM 调用异常也记为拒绝，不中断全量
            r = {"ok": False, "fact": None, "reason": f"exception:{type(e).__name__}"}
        if r["ok"]:
            facts.append(r["fact"])
        else:
            rejects.append({"competition": post.get("competition"),
                            "topic_id": post.get("topic_id"),
                            "reason": r["reason"]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    with rejects_path.open("w", encoding="utf-8") as fh:
        for r in rejects:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(facts) + len(rejects)
    rate = len(rejects) / total if total else 0
    print(f"[extract] facts={len(facts)} rejects={len(rejects)} "
          f"reject_rate={rate:.1%}")
    if rate >= 0.30:
        print("[extract] [WARN] 拒绝率 >= 30%，prompt 或截断策略可能要调")
    return len(facts), len(rejects)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="posts.jsonl → facts.jsonl")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rejects", type=Path, default=DEFAULT_REJECTS)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（试跑）")
    args = ap.parse_args()
    run_extract(args.in_path, args.out, args.rejects, limit=args.limit)


if __name__ == "__main__":
    _cli()
