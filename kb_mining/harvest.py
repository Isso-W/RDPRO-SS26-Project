"""harvest.py — Meta Kaggle dump → data/posts.jsonl。

链路（已由 data/source_check.md 实证，非计划的"预期链路"）：

    Competitions.ForumId  ==  ForumTopics.ForumId
    ForumTopics.FirstForumMessageId  ==  ForumMessages.Id   # 直接外键指向楼主首帖

因此收正文只需：先从 ForumTopics 收一批 FirstForumMessageId，再对 1.7GB 的
ForumMessages **单次 chunksize 流式**按 Id 过滤——不按 ForumTopicId 分组。

核心函数纯粹、可注入、可离线测试；下载/LLM 二级召回是仅有的外部依赖。

CLI:
    python -m kb_mining.harvest [--dump-dir DIR] [--force-download] [--list-recent]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from kb_mining import catalog

# ── 常量 ────────────────────────────────────────────────────────────────────
DEFAULT_DUMP_DIR = Path("kb_mining/data/meta_kaggle")
DEFAULT_OUT = Path("kb_mining/data/posts.jsonl")
META_FILES = ("Competitions.csv", "ForumTopics.csv", "ForumMessages.csv")

MAX_POSTS_PER_COMP = 10
SECOND_TIER_TOPN = 30          # 二级召回：按 Score 取论坛前 N 帖交 LLM 判别
RECALL_KEEP_MIN = 3            # < 此值 → 剔除该竞赛
RECALL_FULL_MIN = 5            # >= 此值 → 正常；[KEEP_MIN, FULL_MIN) → 保留 + warning
FORUM_MSG_CHUNK = 100_000      # ForumMessages 流式 chunk

# solution 帖判定（对 topic 标题）
RANK_RE = re.compile(r"\b(\d{1,3})(st|nd|rd|th)\s+place\b|\bplace\s+(\d{1,3})\b", re.I)
SOLUTION_RE = re.compile(r"solution|write.?up|summary", re.I)


# ── 下载 ────────────────────────────────────────────────────────────────────
def ensure_dump(dump_dir: Path, force: bool = False) -> None:
    """确保三个 Meta Kaggle CSV 就位；缺失（或 force）则用 Kaggle API 下载。"""
    dump_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in META_FILES if not (dump_dir / f).exists()]
    if not missing and not force:
        print(f"[harvest] dump 已就位于 {dump_dir}，跳过下载（--force-download 重取）。")
        return
    try:
        from ingestion.kaggle_loader import _authenticate
    except ImportError:
        # 后台/非仓库根 cwd 下 ingestion 可能不可导入，内联等价实现
        def _authenticate():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            return api
    api = _authenticate()
    targets = META_FILES if force else missing
    for f in targets:
        print(f"[harvest] 下载 {f} ...", flush=True)
        api.dataset_download_file("kaggle/meta-kaggle", f, path=str(dump_dir))
    print("[harvest] 下载完成。")


# ── 纯函数：标题解析 ─────────────────────────────────────────────────────────
def parse_rank(title: str) -> int | None:
    """从 topic 标题解析名次；无名次 → None。"""
    if not title:
        return None
    m = RANK_RE.search(title)
    if not m:
        return None
    num = m.group(1) or m.group(3)
    return int(num) if num else None


def is_solution_title(title: str) -> bool:
    """标题是否像 solution 帖（命中名次 或 solution/writeup/summary）。"""
    if not title:
        return False
    return bool(RANK_RE.search(title) or SOLUTION_RE.search(title))


# ── 竞赛 → ForumId ──────────────────────────────────────────────────────────
def competition_forumids(
    competitions_df: pd.DataFrame,
    competitions: dict[str, dict],
) -> dict[int, str]:
    """{ForumId(int) -> slug}，仅 catalog 里的竞赛。slug 大小写敏感（对齐 dump 原值）。"""
    wanted = {c["slug"] for c in competitions.values()}
    out: dict[int, str] = {}
    for _, r in competitions_df.iterrows():
        if r["Slug"] in wanted and not pd.isna(r["ForumId"]):
            out[int(r["ForumId"])] = r["Slug"]
    return out


# ── ForumTopics → solution 帖记录 ───────────────────────────────────────────
def select_solution_topics(
    topics_df: pd.DataFrame,
    forumid_to_slug: dict[int, str],
) -> list[dict]:
    """筛出属于目标论坛、且标题像 solution 的 topic 记录（含 rank/score/首帖 id）。"""
    records: list[dict] = []
    for _, r in topics_df.iterrows():
        fid = r["ForumId"]
        if pd.isna(fid) or int(fid) not in forumid_to_slug:
            continue
        title = "" if pd.isna(r["Title"]) else str(r["Title"])
        if not is_solution_title(title):
            continue
        if pd.isna(r["FirstForumMessageId"]):
            continue
        records.append({
            "competition": forumid_to_slug[int(fid)],
            "topic_id": int(r["Id"]),
            "topic_title": title,
            "rank": parse_rank(title),
            "first_message_id": int(r["FirstForumMessageId"]),
            "score": 0 if pd.isna(r.get("Score")) else int(r["Score"]),
        })
    return records


def cap_per_competition(
    records: list[dict],
    max_posts: int = MAX_POSTS_PER_COMP,
) -> list[dict]:
    """每竞赛按 (rank 升序, rank None 排后, score 降序) 取前 max_posts 篇。"""
    by_comp: dict[str, list[dict]] = {}
    for rec in records:
        by_comp.setdefault(rec["competition"], []).append(rec)
    out: list[dict] = []
    for comp, recs in by_comp.items():
        recs.sort(key=lambda d: (d["rank"] is None, d["rank"] if d["rank"] is not None else 0,
                                 -d["score"]))
        out.extend(recs[:max_posts])
    return out


# ── ForumMessages 流式取正文 ────────────────────────────────────────────────
def stream_message_bodies(
    messages_path: Path,
    want_ids: set[int],
    chunksize: int = FORUM_MSG_CHUNK,
) -> dict[int, dict]:
    """单次流式扫 ForumMessages，返回 {message_id -> {text, post_date}}。

    正文优先 RawMarkdown，空则回退 Message；post_date 归一为 YYYY-MM-DD。
    """
    if not want_ids:
        return {}
    remaining = set(want_ids)
    out: dict[int, dict] = {}
    for chunk in pd.read_csv(
        messages_path,
        usecols=["Id", "PostDate", "RawMarkdown", "Message"],
        chunksize=chunksize,
    ):
        hit = chunk[chunk["Id"].isin(remaining)]
        for _, r in hit.iterrows():
            raw = r["RawMarkdown"]
            text = raw if isinstance(raw, str) and raw.strip() else r["Message"]
            out[int(r["Id"])] = {
                "text": "" if pd.isna(text) else str(text),
                "post_date": _norm_date(r["PostDate"]),
            }
            remaining.discard(int(r["Id"]))
        if not remaining:
            break
    return out


def _norm_date(raw) -> str | None:
    if pd.isna(raw):
        return None
    try:
        return pd.to_datetime(raw).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


# ── 二级召回（LLM 判别）─────────────────────────────────────────────────────
def second_tier_topics(
    topics_df: pd.DataFrame,
    forum_id: int,
    exclude_topic_ids: set[int],
    top_n: int = SECOND_TIER_TOPN,
) -> list[dict]:
    """某论坛按 Score 取前 top_n 个未入选 topic，供 LLM 判别是否名次方案帖。"""
    sub = topics_df[topics_df["ForumId"] == forum_id].copy()
    sub = sub[~sub["Id"].isin(exclude_topic_ids)]
    sub = sub[~sub["FirstForumMessageId"].isna()]
    sub["Score"] = sub["Score"].fillna(0)
    sub = sub.sort_values("Score", ascending=False).head(top_n)
    return [{
        "topic_id": int(r["Id"]),
        "topic_title": "" if pd.isna(r["Title"]) else str(r["Title"]),
        "first_message_id": int(r["FirstForumMessageId"]),
        "score": int(r["Score"]),
    } for _, r in sub.iterrows()]


# ── 编排 ────────────────────────────────────────────────────────────────────
def run_harvest(
    dump_dir: Path = DEFAULT_DUMP_DIR,
    out_path: Path = DEFAULT_OUT,
    competitions: dict[str, dict] | None = None,
    judge_fn: Callable[[str, str], bool] | None = None,
    chunksize: int = FORUM_MSG_CHUNK,
    keep_min: int = RECALL_KEEP_MIN,
    full_min: int = RECALL_FULL_MIN,
) -> list[dict]:
    """全流程：CSV → 筛帖 → 取正文 → 召回政策 → 写 posts.jsonl。返回 posts。

    judge_fn(title, snippet)->bool 为二级召回的 LLM 判别器；None 则跳过二级召回。
    keep_min / full_min 为召回政策阈值（默认取模块常量）。
    """
    competitions = competitions or catalog.COMPETITIONS
    comp_df = pd.read_csv(dump_dir / "Competitions.csv",
                          usecols=["Slug", "ForumId"])
    topics_df = pd.read_csv(dump_dir / "ForumTopics.csv",
                            usecols=["Id", "ForumId", "FirstForumMessageId",
                                     "Title", "Score"])

    forumid_to_slug = competition_forumids(comp_df, competitions)
    slug_to_forumid = {v: k for k, v in forumid_to_slug.items()}

    primary = select_solution_topics(topics_df, forumid_to_slug)
    capped = cap_per_competition(primary)

    # 二级召回：一级不足 RECALL_FULL_MIN 篇的竞赛
    per_comp: dict[str, list[dict]] = {}
    for rec in capped:
        per_comp.setdefault(rec["competition"], []).append(rec)

    if judge_fn is not None:
        for slug, forum_id in slug_to_forumid.items():
            have = per_comp.get(slug, [])
            if len(have) >= full_min:
                continue
            exclude = {r["topic_id"] for r in have}
            cands = second_tier_topics(topics_df, forum_id, exclude)
            # 取候选正文片段供判别
            snip_ids = {c["first_message_id"] for c in cands}
            snips = stream_message_bodies(dump_dir / "ForumMessages.csv", snip_ids,
                                          chunksize=chunksize)
            for c in cands:
                if len(per_comp.get(slug, [])) >= full_min:
                    break
                body = snips.get(c["first_message_id"], {}).get("text", "")
                if judge_fn(c["topic_title"], body[:500]):
                    c["competition"] = slug
                    c["rank"] = parse_rank(c["topic_title"])
                    per_comp.setdefault(slug, []).append(c)

    # 召回政策：< RECALL_KEEP_MIN 剔除
    kept: list[dict] = []
    dropped: list[str] = []
    warned: list[str] = []
    for slug in slug_to_forumid:
        recs = per_comp.get(slug, [])
        if len(recs) < keep_min:
            dropped.append(f"{slug} ({len(recs)})")
            continue
        if len(recs) < full_min:
            warned.append(f"{slug} ({len(recs)})")
        kept.extend(recs)

    # 取正文
    want_ids = {r["first_message_id"] for r in kept}
    bodies = stream_message_bodies(dump_dir / "ForumMessages.csv", want_ids,
                                   chunksize=chunksize)

    posts: list[dict] = []
    for r in kept:
        body = bodies.get(r["first_message_id"])
        if not body or not body["text"].strip():
            continue
        posts.append({
            "competition": r["competition"],
            "topic_id": r["topic_id"],
            "topic_title": r["topic_title"],
            "rank": r["rank"],
            "author_message_id": r["first_message_id"],
            "text": body["text"],
            "post_date": body["post_date"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in posts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 汇总
    print(f"[harvest] 写出 {len(posts)} 篇到 {out_path}")
    print(f"[harvest] 竞赛保留 {len(slug_to_forumid) - len(dropped)}/{len(slug_to_forumid)}")
    if warned:
        print(f"[harvest] [WARN] 篇数不足 {full_min} 但保留: {', '.join(warned)}")
    if dropped:
        print(f"[harvest] [DROP] 篇数 < {keep_min} 已剔除: {', '.join(dropped)}")
    return posts


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Meta Kaggle dump → posts.jsonl")
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--list-recent", action="store_true",
                    help="只枚举近期 CV 竞赛候选（供人工挑选补进 catalog），不 harvest")
    args = ap.parse_args()

    if args.list_recent:
        ensure_dump(args.dump_dir)  # 只需 Competitions.csv，但一并确保
        for d in catalog.list_recent_cv_candidates(args.dump_dir):
            print(f"{d['teams']:6d}  {d['end']}  {d['slug']:50s} {d['title'][:50]}")
        return

    ensure_dump(args.dump_dir, force=args.force_download)
    run_harvest(args.dump_dir, args.out)


if __name__ == "__main__":
    _cli()
