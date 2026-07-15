"""Harvest solution posts from the Meta Kaggle dump into data/posts.jsonl.

Verified join chain from data/source_check.md:

    Competitions.ForumId  ==  ForumTopics.ForumId
    ForumTopics.FirstForumMessageId  ==  ForumMessages.Id

The text collection path is therefore simple: collect FirstForumMessageId values
from ForumTopics, then stream the 1.7GB ForumMessages CSV once by chunksize and
filter by Id. No ForumTopicId grouping is needed.

Core helpers are pure, injectable, and offline-testable. Downloading and the
optional LLM second-tier recall are the only external dependencies.

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

# Constants.
DEFAULT_DUMP_DIR = Path("kb_mining/data/meta_kaggle")
DEFAULT_OUT = Path("kb_mining/data/posts.jsonl")
META_FILES = ("Competitions.csv", "ForumTopics.csv", "ForumMessages.csv")

MAX_POSTS_PER_COMP = 10
SECOND_TIER_TOPN = 30          # top-scored forum topics checked by LLM recall
RECALL_KEEP_MIN = 3            # below this, drop the competition
RECALL_FULL_MIN = 5            # below this, keep with warning
FORUM_MSG_CHUNK = 100_000      # ForumMessages streaming chunk size

# Solution-post detection from topic titles.
RANK_RE = re.compile(r"\b(\d{1,3})(st|nd|rd|th)\s+place\b|\bplace\s+(\d{1,3})\b", re.I)
SOLUTION_RE = re.compile(r"solution|write.?up|summary", re.I)


# Download.
def ensure_dump(dump_dir: Path, force: bool = False) -> None:
    """Ensure the three Meta Kaggle CSV files exist, downloading if needed."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in META_FILES if not (dump_dir / f).exists()]
    if not missing and not force:
        print(f"[harvest] dump already exists at {dump_dir}; skipping download.")
        return
    try:
        from ingestion.kaggle_loader import _authenticate
    except ImportError:
        # Keep the script usable when ingestion is not importable from cwd.
        def _authenticate():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            return api
    api = _authenticate()
    targets = META_FILES if force else missing
    for f in targets:
        print(f"[harvest] downloading {f} ...", flush=True)
        api.dataset_download_file("kaggle/meta-kaggle", f, path=str(dump_dir))
    print("[harvest] download complete.")


# Pure helper: title parsing.
def parse_rank(title: str) -> int | None:
    """Parse a rank from a topic title, or return None."""
    if not title:
        return None
    m = RANK_RE.search(title)
    if not m:
        return None
    num = m.group(1) or m.group(3)
    return int(num) if num else None


def is_solution_title(title: str) -> bool:
    """Return whether a topic title looks like a solution/writeup post."""
    if not title:
        return False
    return bool(RANK_RE.search(title) or SOLUTION_RE.search(title))


# Competition to ForumId mapping.
def competition_forumids(
    competitions_df: pd.DataFrame,
    competitions: dict[str, dict],
) -> dict[int, str]:
    """Return {ForumId(int): slug} for competitions in the catalog."""
    wanted = {c["slug"] for c in competitions.values()}
    out: dict[int, str] = {}
    for _, r in competitions_df.iterrows():
        if r["Slug"] in wanted and not pd.isna(r["ForumId"]):
            out[int(r["ForumId"])] = r["Slug"]
    return out


# ForumTopics to solution-topic records.
def select_solution_topics(
    topics_df: pd.DataFrame,
    forumid_to_slug: dict[int, str],
) -> list[dict]:
    """Select target-forum topics that look like solution posts."""
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
    """Keep top records per competition by rank and score."""
    by_comp: dict[str, list[dict]] = {}
    for rec in records:
        by_comp.setdefault(rec["competition"], []).append(rec)
    out: list[dict] = []
    for comp, recs in by_comp.items():
        recs.sort(key=lambda d: (d["rank"] is None, d["rank"] if d["rank"] is not None else 0,
                                 -d["score"]))
        out.extend(recs[:max_posts])
    return out


# Stream ForumMessages and collect bodies.
def stream_message_bodies(
    messages_path: Path,
    want_ids: set[int],
    chunksize: int = FORUM_MSG_CHUNK,
) -> dict[int, dict]:
    """Stream ForumMessages once and return {message_id: {text, post_date}}.

    RawMarkdown is preferred; Message is used as fallback. post_date is
    normalized to YYYY-MM-DD.
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


# Second-tier recall, optionally LLM-judged.
def second_tier_topics(
    topics_df: pd.DataFrame,
    forum_id: int,
    exclude_topic_ids: set[int],
    top_n: int = SECOND_TIER_TOPN,
) -> list[dict]:
    """Return top-scored unselected topics for optional LLM judging."""
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


# Orchestration.
def run_harvest(
    dump_dir: Path = DEFAULT_DUMP_DIR,
    out_path: Path = DEFAULT_OUT,
    competitions: dict[str, dict] | None = None,
    judge_fn: Callable[[str, str], bool] | None = None,
    chunksize: int = FORUM_MSG_CHUNK,
    keep_min: int = RECALL_KEEP_MIN,
    full_min: int = RECALL_FULL_MIN,
) -> list[dict]:
    """Run CSV filtering, text collection, recall policy, and posts.jsonl write.

    judge_fn(title, snippet)->bool is the optional second-tier judge. keep_min
    and full_min control the recall policy.
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

    # Second-tier recall for competitions below the full threshold.
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
            # Fetch candidate snippets for judging.
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

    # Recall policy: drop competitions below RECALL_KEEP_MIN.
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

    # Fetch full bodies.
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

    # Summary.
    print(f"[harvest] wrote {len(posts)} posts to {out_path}")
    print(f"[harvest] kept {len(slug_to_forumid) - len(dropped)}/{len(slug_to_forumid)} competitions")
    if warned:
        print(f"[harvest] [WARN] kept with fewer than {full_min} posts: {', '.join(warned)}")
    if dropped:
        print(f"[harvest] [DROP] fewer than {keep_min} posts: {', '.join(dropped)}")
    return posts


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Meta Kaggle dump → posts.jsonl")
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--list-recent", action="store_true",
                    help="List recent CV competition candidates without harvesting")
    args = ap.parse_args()

    if args.list_recent:
        ensure_dump(args.dump_dir)
        for d in catalog.list_recent_cv_candidates(args.dump_dir):
            print(f"{d['teams']:6d}  {d['end']}  {d['slug']:50s} {d['title'][:50]}")
        return

    ensure_dump(args.dump_dir, force=args.force_download)
    run_harvest(args.dump_dir, args.out)


if __name__ == "__main__":
    _cli()
