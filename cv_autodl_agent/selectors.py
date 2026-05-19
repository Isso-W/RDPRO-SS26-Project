from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .exceptions import CandidateSelectionError
from .schemas import DatasetManifest, RetrievedModelCandidate

_BAD_LICENSE_KEYWORDS = ("non-commercial", "restricted", "research only")
_COMPLEXITY_PENALTIES = {
    "huge": 8,
    "giant": 8,
    "large": 5,
    "7b": 7,
    "13b": 8,
    "high memory": 6,
}


@dataclass(slots=True)
class ScoredCandidate:
    candidate: RetrievedModelCandidate
    score: float
    reasons: list[str]



def _score_candidate(manifest: DatasetManifest, candidate: RetrievedModelCandidate) -> ScoredCandidate:
    if candidate.task_family != manifest.task_family:
        return ScoredCandidate(candidate=candidate, score=-inf, reasons=["task_mismatch"])

    score = 100.0 - (candidate.rank * 3)
    reasons = [f"base_rank_bonus:{score}"]

    source = candidate.source.lower()
    library = candidate.library.lower()
    notes = f"{candidate.model_id} {candidate.training_notes or ''}".lower()
    license_name = (candidate.license or "").lower()

    if manifest.task_family == "classification":
        if source in {"timm", "huggingface"}:
            score += 10
            reasons.append("supported_classification_source")
        else:
            score -= 50
            reasons.append("unsupported_classification_source")
    else:
        if source == "huggingface":
            score += 10
            reasons.append("supported_dense_task_source")
        else:
            score -= 50
            reasons.append("unsupported_dense_task_source")

    if "pytorch" in library:
        score += 6
        reasons.append("pytorch_bonus")

    if any(keyword in license_name for keyword in _BAD_LICENSE_KEYWORDS):
        score -= 30
        reasons.append("license_penalty")
    elif license_name:
        score += 4
        reasons.append("license_known_bonus")

    if manifest.image_size_hint and candidate.default_input_size:
        distance = abs(manifest.image_size_hint - candidate.default_input_size)
        if distance <= 32:
            score += 4
            reasons.append("input_size_match")
        elif distance > 128:
            score -= 2
            reasons.append("input_size_mismatch")

    for keyword, penalty in _COMPLEXITY_PENALTIES.items():
        if keyword in notes:
            score -= penalty
            reasons.append(f"complexity_penalty:{keyword}")

    if candidate.install_deps and len(candidate.install_deps) <= 3:
        score += 3
        reasons.append("dependency_bonus")
    elif len(candidate.install_deps) > 5:
        score -= 4
        reasons.append("dependency_penalty")

    return ScoredCandidate(candidate=candidate, score=score, reasons=reasons)



def rank_candidates(manifest: DatasetManifest, candidates: list[RetrievedModelCandidate]) -> list[ScoredCandidate]:
    scored = [_score_candidate(manifest, candidate) for candidate in candidates]
    return sorted(scored, key=lambda item: item.score, reverse=True)



def select_candidate(manifest: DatasetManifest, candidates: list[RetrievedModelCandidate]) -> RetrievedModelCandidate:
    ranked = rank_candidates(manifest, candidates)
    for scored in ranked:
        if scored.score > float("-inf"):
            return scored.candidate
    raise CandidateSelectionError("No candidate matches the provided manifest")
