"""Model soup — average the weights of K finetuned variants into ONE model.

Goal: ensemble-like accuracy at single-model inference cost. Under a strict
single-model deployment budget this is an edge a multi-model ensemble can't use
(the ensemble violates the budget; the soup is still one model).

Greedy soup (Wortsman et al., 2022): rank variants by validation score, then add a
variant to the soup only if it improves the running validation score.

Two entry points:
  # soup checkpoints you already trained
  python soup.py greedy --project run/module4_code --checkpoints a.pt b.pt c.pt
  # train K variants (different seeds) then soup them
  python soup.py run --project run/module4_code --n 5 --epochs 15 --dataset uoft-cs/cifar10

This validates Phase 0: does the soup actually beat the best single model?
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── checkpoint / weight-averaging primitives ───────────────────────────────────

def _load_state(path: str | Path) -> dict:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def _average_states(states: list[dict]) -> dict:
    """Element-wise mean of float tensors; non-float buffers keep the first value."""
    import torch

    averaged: dict = {}
    for key in states[0]:
        values = [s[key] for s in states]
        first = values[0]
        if torch.is_floating_point(first):
            stacked = torch.stack([v.float() for v in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(first.dtype)
        else:
            averaged[key] = first  # e.g. num_batches_tracked — averaging is meaningless
    return averaged


# ── evaluation in the generated-project context ────────────────────────────────

def _flatten_config(config: dict) -> dict:
    merged = dict(config)
    mc = config.get("model_config")
    if isinstance(mc, dict):
        for k, v in mc.items():
            if v is not None or k not in merged:
                merged[k] = v
    return merged


def _project_config(project_dir: Path) -> dict:
    return _flatten_config(json.loads((project_dir / "configs.json").read_text(encoding="utf-8"))[0])


def _eval_state(project_dir: Path, config: dict, state: dict) -> dict:
    """Build the project's model, load `state`, and return its evaluate() result."""
    import torch

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    cwd = os.getcwd()
    os.chdir(project_dir)
    try:
        from model import build_model
        from evaluate import evaluate

        model = build_model(config)
        model.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return evaluate(model, config)
    finally:
        os.chdir(cwd)


def _metric(result: dict) -> float:
    return float(result.get("metric_value", result.get("accuracy", 0.0)))


# ── greedy soup ────────────────────────────────────────────────────────────────

def greedy_soup(project_dir: str | Path, checkpoint_paths: list[str], minimize: bool | None = None):
    """Greedily build a soup from the given checkpoints. Returns (souped_state, report)."""
    project_dir = Path(project_dir).resolve()
    config = _project_config(project_dir)
    if minimize is None:
        metric_name = str(config.get("evaluation_metric", "accuracy")).lower()
        minimize = metric_name in {"log_loss", "multiclass_log_loss", "rmse"}

    # Score each variant individually.
    scored = []
    for path in checkpoint_paths:
        state = _load_state(path)
        score = _metric(_eval_state(project_dir, config, state))
        scored.append({"path": str(path), "state": state, "score": score})
        print(f"[soup] variant {Path(path).name}: score={score:.4f}")

    scored.sort(key=lambda d: d["score"], reverse=not minimize)
    best_single = scored[0]["score"]

    soup_states = [scored[0]["state"]]
    running = best_single
    log = [{"action": "start", "path": scored[0]["path"], "score": running}]

    for cand in scored[1:]:
        trial = _average_states(soup_states + [cand["state"]])
        trial_score = _metric(_eval_state(project_dir, config, trial))
        improved = (trial_score < running) if minimize else (trial_score > running)
        if improved:
            soup_states.append(cand["state"])
            running = trial_score
            log.append({"action": "add", "path": cand["path"], "score": trial_score})
            print(f"[soup] + {Path(cand['path']).name}: soup score -> {trial_score:.4f}")
        else:
            log.append({"action": "skip", "path": cand["path"], "score": trial_score})
            print(f"[soup] - {Path(cand['path']).name}: would be {trial_score:.4f} (no improvement)")

    souped = _average_states(soup_states)
    report = {
        "n_candidates": len(checkpoint_paths),
        "n_in_soup": len(soup_states),
        "best_single": best_single,
        "soup_score": running,
        "improvement": running - best_single if not minimize else best_single - running,
        "minimize": minimize,
        "log": log,
    }
    return souped, report


def save_soup(souped_state: dict, out_path: str | Path) -> None:
    import torch

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": souped_state, "soup": True}, out_path)
    print(f"[soup] Saved souped model -> {out_path}")


# ── train K variants, then soup ────────────────────────────────────────────────

def train_variants(project_dir: str | Path, n: int, epochs: int, dataset: str | None = None) -> list[str]:
    """Train n variants (different seeds) into separate checkpoint dirs. Returns best_model.pt paths."""
    project_dir = Path(project_dir).resolve()
    cfg_path = project_dir / "configs.json"
    base = json.loads(cfg_path.read_text(encoding="utf-8"))
    paths = []
    for i in range(n):
        ckpt_dir = project_dir / "soup_variants" / f"v{i}"
        cfg = json.loads(json.dumps(base))  # deep copy
        cfg[0]["checkpoint_dir"] = str(ckpt_dir)
        cfg[0]["seed"] = 1000 + i
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = [sys.executable, "-u", "run.py", "--epochs", str(epochs), "--seed", str(1000 + i)]
        if dataset:
            cmd += ["--dataset", dataset]
        print(f"\n[soup] === training variant {i + 1}/{n} (seed={1000 + i}) ===")
        subprocess.run(cmd, cwd=str(project_dir), text=True).check_returncode()
        paths.append(str(ckpt_dir / "best_model.pt"))
    cfg_path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")  # restore
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Model soup over finetuned variants.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("greedy", help="Greedy-soup existing checkpoints.")
    g.add_argument("--project", required=True)
    g.add_argument("--checkpoints", nargs="+", required=True)
    g.add_argument("--out", default=None)

    r = sub.add_parser("run", help="Train n variants then greedy-soup them.")
    r.add_argument("--project", required=True)
    r.add_argument("--n", type=int, default=5)
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--dataset", default=None)
    r.add_argument("--out", default=None)

    args = parser.parse_args()
    project = Path(args.project).resolve()

    if args.cmd == "run":
        checkpoints = train_variants(project, args.n, args.epochs, args.dataset)
    else:
        checkpoints = args.checkpoints

    souped, report = greedy_soup(project, checkpoints)
    out = Path(args.out) if args.out else project / "checkpoints" / "soup_model.pt"
    save_soup(souped, out)

    print("\n" + "=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 60)
    verdict = "SOUP WINS" if report["improvement"] > 0 else "no gain"
    print(f"best_single={report['best_single']:.4f}  soup={report['soup_score']:.4f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
