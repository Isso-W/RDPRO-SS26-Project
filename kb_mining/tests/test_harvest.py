"""test_harvest.py — 全离线，用迷你 CSV fixtures 验证 harvest 的纯函数与编排。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kb_mining import harvest

FIX = Path(__file__).parent / "fixtures"

# catalog 的子集：只含 alpha/beta；comp-unknown 故意不在 → 应被 slug 过滤掉
TEST_COMPS = {
    "comp-alpha": {"slug": "comp-alpha"},
    "comp-beta": {"slug": "comp-beta"},
}


# ── 纯函数：标题解析 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("title,expected", [
    ("1st Place Solution", 1),
    ("2nd place solution", 2),
    ("3rd place writeup", 3),
    ("14th Place Solution & Code", 14),
    ("10th place solution (23rd public)", 10),
    ("Private LB place 5 silver", 5),
    ("Solution summary", None),
    ("Random chat thread", None),
    ("", None),
])
def test_parse_rank(title, expected):
    assert harvest.parse_rank(title) == expected


@pytest.mark.parametrize("title,expected", [
    ("1st Place Solution", True),
    ("Solution summary", True),
    ("my writeup here", True),
    ("2nd place", True),
    ("Random chat thread", False),
    ("Welcome to the competition", False),
    ("", False),
])
def test_is_solution_title(title, expected):
    assert harvest.is_solution_title(title) is expected


# ── 竞赛 → ForumId（slug 过滤）───────────────────────────────────────────────
def test_competition_forumids_filters_by_slug():
    df = pd.read_csv(FIX / "Competitions.csv", usecols=["Slug", "ForumId"])
    m = harvest.competition_forumids(df, TEST_COMPS)
    assert m == {100: "comp-alpha", 200: "comp-beta"}   # 300/comp-unknown 被过滤


# ── ForumTopics → solution 帖 ───────────────────────────────────────────────
def test_select_solution_topics():
    df = pd.read_csv(FIX / "ForumTopics.csv")
    recs = harvest.select_solution_topics(df, {100: "comp-alpha", 200: "comp-beta"})
    # t4(chat) 不匹配正则、t6(forum 300 不在 map) 被排除 → 剩 4 篇
    assert len(recs) == 4
    by_id = {r["topic_id"]: r for r in recs}
    assert set(by_id) == {1, 2, 3, 5}
    assert by_id[1]["rank"] == 1 and by_id[1]["competition"] == "comp-alpha"
    assert by_id[3]["rank"] is None            # "Solution summary" 无名次
    assert by_id[5]["competition"] == "comp-beta"
    assert by_id[1]["first_message_id"] == 1001


def test_cap_per_competition_orders_and_caps():
    df = pd.read_csv(FIX / "ForumTopics.csv")
    recs = harvest.select_solution_topics(df, {100: "comp-alpha", 200: "comp-beta"})
    capped = harvest.cap_per_competition(recs, max_posts=2)
    alpha = [r for r in capped if r["competition"] == "comp-alpha"]
    beta = [r for r in capped if r["competition"] == "comp-beta"]
    assert len(alpha) == 2                      # 截断到 2
    assert [r["rank"] for r in alpha] == [1, 2]  # 有名次的排前，summary(None) 被挤出
    assert len(beta) == 1


# ── ForumMessages 流式取正文（chunksize=2 强制多 chunk）──────────────────────
def test_stream_message_bodies_chunked():
    got = harvest.stream_message_bodies(
        FIX / "ForumMessages.csv",
        want_ids={1001, 1002, 1005},
        chunksize=2,                             # 6 行 → 3 个 chunk，强制多 chunk 路径
    )
    assert set(got) == {1001, 1002, 1005}
    assert got[1001]["text"] == "alpha winner body"          # RawMarkdown 优先
    assert got[1002]["text"] == "<b>alpha 2nd body</b>"      # RawMarkdown 空 → Message 回退
    assert got[1001]["post_date"] == "2021-02-24"            # 日期归一


def test_stream_message_bodies_empty():
    assert harvest.stream_message_bodies(FIX / "ForumMessages.csv", set()) == {}


# ── 编排 + 召回政策 ─────────────────────────────────────────────────────────
def test_run_harvest_recall_drops_thin_competition(tmp_path):
    out = tmp_path / "posts.jsonl"
    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=out, competitions=TEST_COMPS,
        chunksize=2, keep_min=3, full_min=5,
    )
    # alpha 有 3 篇(t1,t2,t3) → 保留；beta 仅 1 篇 → 剔除
    comps = {p["competition"] for p in posts}
    assert comps == {"comp-alpha"}
    assert len(posts) == 3
    assert out.exists()
    # 落盘内容与返回一致
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_run_harvest_low_threshold_keeps_beta(tmp_path):
    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=tmp_path / "posts.jsonl", competitions=TEST_COMPS,
        chunksize=2, keep_min=1, full_min=5,
    )
    comps = {p["competition"] for p in posts}
    assert comps == {"comp-alpha", "comp-beta"}   # beta(1 篇) 现在也保留
    assert len(posts) == 4


def test_run_harvest_second_tier_recall(tmp_path):
    # judge_fn 把 "Random chat thread"(t4) 也判为方案帖 → beta 之外, alpha 已够
    # 这里验证二级召回能把非正则命中的帖补进来：用 keep_min=1 让 beta 参与，
    # judge_fn 对 beta 论坛的 chat 帖返回 True。beta 论坛无额外帖，故构造 alpha
    # 的场景：把 full_min 设高触发二级召回，judge 认领 t4。
    def judge(title, snippet):
        return "chat" in title.lower()

    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=tmp_path / "posts.jsonl", competitions=TEST_COMPS,
        judge_fn=judge, chunksize=2, keep_min=1, full_min=10,
    )
    alpha_titles = {p["topic_title"] for p in posts if p["competition"] == "comp-alpha"}
    assert "Random chat thread" in alpha_titles   # 二级召回补入
