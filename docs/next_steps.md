# Next steps / open decisions

Planning notes for the accumulating-recommender direction (branch:
`integration-recommender`). The thesis: occupy the corner MLE-STAR structurally can't —
**cheap, explainable, constraint-aware, accumulating** model recommendation — rather than
out-searching it.

## Open decision: ensembles (needs discussion)

Transformer/foundation-model era rarely uses explicit ensembles, because finetunes from the
**same pretrained checkpoint are highly correlated** → little diversity → little ensemble
(and model-soup) gain. Remaining ensemble value is in **cross-architecture / cross-pretraining**
diversity, which Module 3's heterogeneous candidate pool already provides.

Three options (undecided):

- **A. Build ensembles as a niche feature** — but framed as *cross-architecture complementary*
  ensembles (greedy selection over KB candidates of different family/pretraining, diversity from
  KB metadata + cheap LogME/probe predictions). Under a single-model deployment budget, collapse
  same-arch members into a soup.
- **B. Don't reinvest in ensembles** — focus on strong single-model selection + near-free recipe
  upgrades (EMA/SWA) + constraint-awareness. Follows where the field is going.
- **C. Ensemble knowledge as a selection rule** — Jiaozi decides *whether a task is worth
  ensembling* (small data / CNN / generous budget → recommend cross-arch ensemble; big model /
  tight budget → single strong model + EMA). The "judgement" itself is the product.

Leaning: **C + B** (Jiaozi knows when to ensemble; defaults to single strong model + EMA).
Model soup is demoted to an optional product of the ensemble mode under tight deployment budgets,
not a headline.

## Recipe layer (hyperparameters)

- **v0 — done** (`recommender/recipe.py`, pipeline `--use-recipe`). Rules: backbone hard
  constraints (DINOv2 patch-14 image_size) + finetuning conventions (lr by strategy/family
  with backbone_lr_scale, augmentation by data size, early stopping). Opt-in; emits only keys
  the train template consumes (learning_rate, backbone_lr_scale, image_size, augmentation,
  early_stopping_patience).
- **v1 — reserved** (`_llm_recipe_proposal` stub): an LLM proposes the soft HPs grounded by the
  outcome memory, validated against the rule recipe's ranges (rules are the guardrailed floor).
  One call, not a search loop. Expectation: a strong "practitioner first-guess" config, improving
  via memory — measure it with the AIDE harness (rules vs LLM-recipe vs AIDE, quality-vs-cost).
- **later**: weight_decay / warmup once the train template consumes them.

## Pending build items

1. **EMA in generated training** (high value, low risk, no cost tension) — add exponential weight
   averaging to the Module 4 train template: single run, single model, near-free quality gain.
   Strictly better than model soup on the cost axis.
2. **Wire LogME to real data** — `logme_score` is a pure metric; extract frozen features via
   `model_utils` on a data sample in the pipeline to produce `{backbone: logme}` for the cold-start
   ranker (and, later, per-model val predictions for ensemble selection).
3. **Cold-start seeding** — seed the outcome memory from public transfer benchmarks (timm / VTAB)
   or a few real `run_and_log` runs.
4. **AIDE comparison harness** — run AIDE on the same tasks under the same budget, log its quality
   + cost into the same table → the quality-vs-cost Pareto plot (the selling-point figure).
5. **Automated KB updating** (companion pillar) — ingest new backbones/checkpoints from HF /
   papers-with-code into the persistent structured graph, so the recommender isn't capped by a
   narrow hand-curated KB.

## Validation still pending
- Recommender "improves as used" — needs real logged runs.
- Model soup Phase 0 — does it actually beat the best single on real data? (branch `feature/model-soup`)
- Constraint-aware selection vs ground truth (HW-NAS-Bench style) / vs AIDE.
