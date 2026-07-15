"""Offline tests for harvest.py using small CSV fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kb_mining import harvest

FIX = Path(__file__).parent / "fixtures"

# Catalog subset contains alpha/beta only; comp-unknown should be filtered out.
TEST_COMPS = {
    "comp-alpha": {"slug": "comp-alpha"},
    "comp-beta": {"slug": "comp-beta"},
}


# Pure helper: title parsing.
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


# Competition to ForumId mapping.
def test_competition_forumids_filters_by_slug():
    df = pd.read_csv(FIX / "Competitions.csv", usecols=["Slug", "ForumId"])
    m = harvest.competition_forumids(df, TEST_COMPS)
    assert m == {100: "comp-alpha", 200: "comp-beta"}


# ForumTopics to solution-topic records.
def test_select_solution_topics():
    df = pd.read_csv(FIX / "ForumTopics.csv")
    recs = harvest.select_solution_topics(df, {100: "comp-alpha", 200: "comp-beta"})
    # t4(chat) misses the regex; t6 belongs to an unmapped forum.
    assert len(recs) == 4
    by_id = {r["topic_id"]: r for r in recs}
    assert set(by_id) == {1, 2, 3, 5}
    assert by_id[1]["rank"] == 1 and by_id[1]["competition"] == "comp-alpha"
    assert by_id[3]["rank"] is None
    assert by_id[5]["competition"] == "comp-beta"
    assert by_id[1]["first_message_id"] == 1001


def test_cap_per_competition_orders_and_caps():
    df = pd.read_csv(FIX / "ForumTopics.csv")
    recs = harvest.select_solution_topics(df, {100: "comp-alpha", 200: "comp-beta"})
    capped = harvest.cap_per_competition(recs, max_posts=2)
    alpha = [r for r in capped if r["competition"] == "comp-alpha"]
    beta = [r for r in capped if r["competition"] == "comp-beta"]
    assert len(alpha) == 2
    assert [r["rank"] for r in alpha] == [1, 2]
    assert len(beta) == 1


# ForumMessages body streaming.
def test_stream_message_bodies_chunked():
    got = harvest.stream_message_bodies(
        FIX / "ForumMessages.csv",
        want_ids={1001, 1002, 1005},
        chunksize=2,
    )
    assert set(got) == {1001, 1002, 1005}
    assert got[1001]["text"] == "alpha winner body"
    assert got[1002]["text"] == "<b>alpha 2nd body</b>"
    assert got[1001]["post_date"] == "2021-02-24"


def test_stream_message_bodies_empty():
    assert harvest.stream_message_bodies(FIX / "ForumMessages.csv", set()) == {}


# Orchestration and recall policy.
def test_run_harvest_recall_drops_thin_competition(tmp_path):
    out = tmp_path / "posts.jsonl"
    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=out, competitions=TEST_COMPS,
        chunksize=2, keep_min=3, full_min=5,
    )
    # alpha has three posts and is kept; beta has one and is dropped.
    comps = {p["competition"] for p in posts}
    assert comps == {"comp-alpha"}
    assert len(posts) == 3
    assert out.exists()
    # Written content matches returned posts.
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_run_harvest_low_threshold_keeps_beta(tmp_path):
    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=tmp_path / "posts.jsonl", competitions=TEST_COMPS,
        chunksize=2, keep_min=1, full_min=5,
    )
    comps = {p["competition"] for p in posts}
    assert comps == {"comp-alpha", "comp-beta"}
    assert len(posts) == 4


def test_run_harvest_second_tier_recall(tmp_path):
    # Verify that second-tier recall can add a non-regex topic when the judge
    # accepts it.
    def judge(title, snippet):
        return "chat" in title.lower()

    posts = harvest.run_harvest(
        dump_dir=FIX, out_path=tmp_path / "posts.jsonl", competitions=TEST_COMPS,
        judge_fn=judge, chunksize=2, keep_min=1, full_min=10,
    )
    alpha_titles = {p["topic_title"] for p in posts if p["competition"] == "comp-alpha"}
    assert "Random chat thread" in alpha_titles
