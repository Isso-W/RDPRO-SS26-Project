"""Offline tests for extract.py with canned LLM responses."""

from __future__ import annotations

import json
from pathlib import Path

from kb_mining import extract


def canned(payload):
    """Return an llm_fn that ignores inputs and emits the fixed payload."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda system, user: text


# Happy paths.
def test_single_model():
    post = {"competition": "c", "topic_id": 1,
            "text": "We used a single tf_efficientnet_b4 model at 512px."}
    resp = {"kind": "single",
            "members": [{"raw_model": "tf_efficientnet_b4", "image_size": 512}],
            "loss_raw": "focal loss", "best_single_model_raw": None,
            "best_single_score": None, "used_pseudo_labeling": False,
            "used_tta": True, "citations": ["single tf_efficientnet_b4 model"]}
    fact = extract.extract_post(post, canned(resp))
    assert fact is not None
    assert fact["kind"] == "single"
    assert fact["families"] == ["efficientnet"]
    assert fact["family_image_size"] == {"efficientnet": 512}
    assert fact["loss_kb"] == "focal_loss"
    assert "text" not in fact
    assert fact["competition"] == "c"


def test_ensemble_with_best_single():
    post = {"competition": "c", "topic_id": 2,
            "text": "Ensemble of tf_efficientnet_b4, tf_efficientnet_b5 and resnet50. "
                    "Our best single model was the b4."}
    resp = {"kind": "ensemble",
            "members": [{"raw_model": "tf_efficientnet_b4", "image_size": 512},
                        {"raw_model": "tf_efficientnet_b5", "image_size": 512},
                        {"raw_model": "resnet50", "image_size": 384}],
            "loss_raw": "cross entropy", "best_single_model_raw": "tf_efficientnet_b4",
            "best_single_score": 0.899, "used_pseudo_labeling": False,
            "used_tta": False, "citations": ["Ensemble of tf_efficientnet_b4"]}
    fact = extract.extract_post(post, canned(resp))
    assert fact["families"] == ["efficientnet", "resnet"]
    assert fact["family_image_size"]["efficientnet"] == 512
    assert fact["best_single_family"] == "efficientnet"
    assert fact["loss_kb"] == "cross_entropy_loss"


def test_family_dedup_image_size_mode():
    post = {"competition": "c", "topic_id": 3, "text": "b4 512, b5 512, b5 768"}
    resp = {"kind": "ensemble",
            "members": [{"raw_model": "efficientnet_b4", "image_size": 512},
                        {"raw_model": "efficientnet_b5", "image_size": 512},
                        {"raw_model": "efficientnet_b5", "image_size": 768}],
            "loss_raw": None, "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False,
            "citations": ["b4 512, b5 512, b5 768"]}
    fact = extract.extract_post(post, canned(resp))
    assert fact["families"] == ["efficientnet"]
    assert fact["family_image_size"]["efficientnet"] == 512


# Alias mapping.
def test_alias_mapping():
    post = {"competition": "c", "topic_id": 4,
            "text": "tf_efficientnetv2_m and seresnext50 and some_weird_net"}
    resp = {"kind": "ensemble",
            "members": [{"raw_model": "tf_efficientnetv2_m", "image_size": None},
                        {"raw_model": "seresnext50", "image_size": None},
                        {"raw_model": "some_weird_net", "image_size": None}],
            "loss_raw": "arcface", "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False,
            "citations": ["tf_efficientnetv2_m and seresnext50"]}
    fact = extract.extract_post(post, canned(resp))
    assert fact["families"] == ["efficientnet", "resnet", "unknown"]
    assert fact["loss_kb"] == "unknown"
    assert fact["loss_is_metric_learning"] is True


# Rejection paths.
def test_hybrid_loss_maps_to_unknown():
    post = {"competition": "c", "topic_id": 10,
            "text": "loss was Average of BCE and FocalLoss for the heads"}
    resp = {"kind": "single",
            "members": [{"raw_model": "resnet50", "image_size": None}],
            "loss_raw": "Average of BCE and FocalLoss", "best_single_model_raw": None,
            "best_single_score": None, "used_pseudo_labeling": False, "used_tta": False,
            "citations": ["Average of BCE and FocalLoss"]}
    fact = extract.extract_post(post, canned(resp))
    assert fact["loss_kb"] == "unknown"


def test_reject_citation_not_found():
    post = {"competition": "c", "topic_id": 5, "text": "real body text about models"}
    resp = {"kind": "single",
            "members": [{"raw_model": "resnet50", "image_size": None}],
            "loss_raw": None, "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False,
            "citations": ["a hallucinated quote not in the post"]}
    r = extract.classify_post(post, canned(resp))
    assert r["ok"] is False and r["reason"] == "citation_not_found"
    assert extract.extract_post(post, canned(resp)) is None


def test_reject_bad_json():
    post = {"competition": "c", "topic_id": 6, "text": "body"}
    r = extract.classify_post(post, canned("this is not json {{{"))
    assert r["ok"] is False and r["reason"] == "json_parse"


def test_reject_empty_members_non_solution():
    post = {"competition": "c", "topic_id": 7, "text": "Congrats everyone!"}
    resp = {"kind": "unclear", "members": [], "loss_raw": None,
            "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False, "citations": []}
    r = extract.classify_post(post, canned(resp))
    assert r["ok"] is False and r["reason"] == "empty_members"


def test_reject_too_many_members():
    post = {"competition": "c", "topic_id": 8, "text": "x" * 20}
    resp = {"kind": "ensemble",
            "members": [{"raw_model": f"resnet{i}", "image_size": None} for i in range(13)],
            "loss_raw": None, "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False, "citations": ["x"]}
    r = extract.classify_post(post, canned(resp))
    assert r["ok"] is False and r["reason"] == "too_many_members"


def test_strip_json_fences():
    post = {"competition": "c", "topic_id": 9, "text": "resnet50 body"}
    fenced = "```json\n" + json.dumps({
        "kind": "single", "members": [{"raw_model": "resnet50", "image_size": None}],
        "loss_raw": None, "best_single_model_raw": None, "best_single_score": None,
        "used_pseudo_labeling": False, "used_tta": False, "citations": ["resnet50 body"]}) + "\n```"
    fact = extract.extract_post(post, canned(fenced))
    assert fact is not None and fact["families"] == ["resnet"]


def test_truncate_long_text():
    long = "A" * 10000 + "MIDDLE" + "B" * 10000
    out = extract.truncate_text(long)
    assert len(out) < len(long)
    assert out.startswith("A")
    assert out.endswith("B")
    assert "MIDDLE" not in out


# Orchestration writes facts and rejects.
def test_run_extract_writes_files(tmp_path):
    posts = [
        {"competition": "c", "topic_id": 1, "text": "single resnet50 model here"},
        {"competition": "c", "topic_id": 2, "text": "bad one"},
    ]
    in_path = tmp_path / "posts.jsonl"
    in_path.write_text("\n".join(json.dumps(p) for p in posts), encoding="utf-8")

    good = {"kind": "single",
            "members": [{"raw_model": "resnet50", "image_size": None}],
            "loss_raw": None, "best_single_model_raw": None, "best_single_score": None,
            "used_pseudo_labeling": False, "used_tta": False,
            "citations": ["single resnet50 model"]}

    # First post passes; second post fails citation matching.
    def llm(system, user):
        return json.dumps(good) if "resnet50" in user else json.dumps(
            {**good, "citations": ["nope"]})

    out = tmp_path / "facts.jsonl"
    rej = tmp_path / "rejects.jsonl"
    nf, nr = extract.run_extract(in_path, out, rej, llm_fn=llm)
    assert (nf, nr) == (1, 1)
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1
    rejrec = json.loads(rej.read_text(encoding="utf-8").strip())
    assert rejrec["reason"] == "citation_not_found" and rejrec["topic_id"] == 2
