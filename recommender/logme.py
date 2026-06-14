"""LogME — a near-zero-cost transferability score for the cold-start ranking slot.

LogME (You et al., ICML 2021) estimates how well a backbone's *frozen* features predict
the target labels — a proxy for finetuning quality — without any training. We use it to
rank candidates the outcome memory has no track record for (cold start), in place of the
weak KB heuristic.

`logme_score(features, labels)` is the pure metric (numpy only). Extracting the features
(load backbone, forward over a data sample) is the caller's job — heavy, needs torch + the
dataset — and produces the per-candidate scores fed to the ranker.
"""

from __future__ import annotations

import numpy as np


def _evidence_for_target(y: np.ndarray, sigma: np.ndarray, z: np.ndarray,
                         n: int, d: int, iters: int = 50) -> float:
    """Log maximum evidence of a linear model for one (one-vs-rest) target vector.

    sigma: squared singular values of the feature matrix (length r).
    z:     projection of y onto the left singular vectors (length r), i.e. U^T y.
    """
    # energy of y outside the column span (captured by the residual term)
    y_energy = float((y * y).sum())
    z2 = z * z
    delta = max(y_energy - float(z2.sum()), 0.0)  # residual energy in the null space

    alpha, beta = 1.0, 1.0
    for _ in range(iters):
        ab = alpha + beta * sigma
        gamma = float((beta * sigma / ab).sum())
        m2 = float(((beta ** 2) * sigma * z2 / (ab ** 2)).sum())
        res = float(((alpha ** 2) * z2 / (ab ** 2)).sum()) + delta  # residual sum of squares
        new_alpha = gamma / (m2 + 1e-12)
        new_beta = (n - gamma) / (res + 1e-12)
        if abs(new_alpha - alpha) / alpha < 1e-3 and abs(new_beta - beta) / beta < 1e-3:
            alpha, beta = new_alpha, new_beta
            break
        alpha, beta = new_alpha, new_beta

    ab = alpha + beta * sigma
    m2 = float(((beta ** 2) * sigma * z2 / (ab ** 2)).sum())
    res = float(((alpha ** 2) * z2 / (ab ** 2)).sum()) + delta
    evidence = 0.5 * (
        d * np.log(alpha)
        + n * np.log(beta)
        - float(np.log(ab).sum())
        - beta * res
        - alpha * m2
        - n * np.log(2 * np.pi)
    )
    return float(evidence / n)


def logme_score(features: np.ndarray, labels: np.ndarray) -> float:
    """LogME transferability score: higher means the frozen features fit the labels better.

    features: (N, D) frozen-feature matrix. labels: (N,) integer class labels.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).ravel()
    if f.ndim != 2 or f.shape[0] != y.shape[0] or f.shape[0] < 2:
        raise ValueError("features must be (N, D) and align with N labels (N >= 2).")

    f = f - f.mean(axis=0, keepdims=True)  # centre
    n, d = f.shape
    u, s, _ = np.linalg.svd(f, full_matrices=False)  # u: (N, r), s: (r,)
    sigma = s ** 2

    evidences = []
    for c in np.unique(y):
        target = (y == c).astype(np.float64)
        target = target - target.mean()      # centre the one-vs-rest target
        z = u.T @ target
        evidences.append(_evidence_for_target(target, sigma, z, n, d))
    return float(np.mean(evidences)) if evidences else 0.0
