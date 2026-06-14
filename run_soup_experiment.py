"""One-shot model-soup experiment: dataset -> project -> K variants -> greedy soup.

Validates Phase 0: on a real dataset, does a greedy soup of K finetuned variants
beat the best single model? (Ensemble-like accuracy at single-model inference cost.)

It reuses the full pipeline to generate a real-training project (Module 1->2->3->4),
then trains K seeds and soups them with soup.py.

Prereqs (same as the normal pipeline):
  - an LLM key for Module 1 (e.g. JIAOZI_LLM_PROVIDER=openai + OPENAI_API_KEY[/OPENAI_BASE_URL]),
  - a GPU for real training.

Run (locally or in Colab via `!python run_soup_experiment.py ...`):
  python run_soup_experiment.py \
    --dataset dpdl-benchmark/cassava \
    --query "Classify cassava leaf images, balance accuracy and speed" \
    --n 4 --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end model-soup experiment.")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset id (e.g. dpdl-benchmark/cassava).")
    parser.add_argument("--query", default="Classify images, balancing accuracy and training speed.",
                        help="Natural-language task for Module 1 (keep it neutral; don't name a model).")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name.")
    parser.add_argument("--output", default="./soup_run", help="Where to generate the project.")
    parser.add_argument("--n", type=int, default=4, help="Number of variants to soup.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per variant.")
    parser.add_argument("--llm-provider", default=None,
                        choices=["none", "qwen", "openai", "vertex"],
                        help="Module 4 model.py provider; default env/template.")
    args = parser.parse_args()

    from pipeline import parse_dataset_id, run_pipeline

    dataset_id, parsed_subset = parse_dataset_id(args.dataset)
    subset = args.subset or parsed_subset
    output = Path(args.output).resolve()

    # 1) Generate a real-training project through the full pipeline (Module 1->2->3->4).
    print("=" * 70)
    print("[soup-exp] Step 1/3 — generating a real-training project via the pipeline ...")
    print("=" * 70)
    run_pipeline(
        args.query,
        dataset_id,
        fmt="nl",
        subset=subset,
        module4_output=output,
        module4_real_training=True,     # offline_smoke=false; auto-skips local smoke
        module4_skip_smoke=True,
        module4_llm_provider=args.llm_provider,
    )
    project = output / "module4_code"
    cfg = json.loads((project / "configs.json").read_text(encoding="utf-8"))[0]
    mc = cfg.get("model_config", cfg)
    backbone = mc.get("backbone") or cfg.get("backbone")
    print(f"[soup-exp] Module 3 picked backbone: {backbone}")

    # 2) Train K variants (different seeds).
    print("\n" + "=" * 70)
    print(f"[soup-exp] Step 2/3 — training {args.n} variants x {args.epochs} epochs ...")
    print("=" * 70)
    import soup

    checkpoints = soup.train_variants(project, args.n, args.epochs, dataset_id)

    # 3) Greedy soup + verdict.
    print("\n" + "=" * 70)
    print("[soup-exp] Step 3/3 — greedy soup ...")
    print("=" * 70)
    souped, report = soup.greedy_soup(project, checkpoints)
    soup.save_soup(souped, project / "checkpoints" / "soup_model.pt")

    print("\n" + "#" * 70)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    verdict = "SOUP WINS" if report["improvement"] > 0 else "no gain"
    print(f"\nbackbone={backbone}  variants={args.n}  in_soup={report['n_in_soup']}")
    print(f"best_single={report['best_single']:.4f}   soup={report['soup_score']:.4f}   -> {verdict}")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
