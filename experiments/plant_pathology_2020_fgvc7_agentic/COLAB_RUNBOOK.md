# Plant Pathology 2020 FGVC7 Colab Runbook

This folder was generated locally with Jiaozi plus the agentic Kaggle workflow.
It is intended to be pushed to GitHub, cloned on Colab, and trained with a GPU.

## Competition Contract

- Kaggle slug: `plant-pathology-2020-fgvc7`
- Submission mode: classic `submission.csv`
- Target columns: `healthy`, `multiple_diseases`, `rust`, `scab`
- Local validation: stratified folds over the materialized single-label column
- Metric used in generated code: `roc_auc`, matching Kaggle's mean per-column AUC intent

## Colab Commands

Clone your branch:

```bash
git clone -b codex/plant-pathology-producer-consumer-ensemble https://github.com/Isso-W/Jiaozi.git
cd Jiaozi
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install --upgrade "kaggle>=2.2.2" transformers datasets timm scikit-learn
```

Provide Kaggle credentials in Colab, then accept the competition rules once in
the Kaggle UI. Use `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME` + `KAGGLE_KEY`, or
`KAGGLE_JSON` containing the downloaded `kaggle.json` payload.

Prepare data, one-hot labels, folds, and patched configs:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py
```

Train the first generated config, DINOv3-B partial last2:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --train --epochs 5
```

Train every generated candidate with the same fold:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --train-all --epochs 5
```

## Producer / Consumer Ensemble

Train all producer candidates for one fold and export `oof.csv`,
`test_probs.csv`, checkpoints, and `producer_manifest.json`:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py \
  --fold-index 0 \
  --train-producers \
  --epochs 5
```

For a proper OOF blend, run all folds with the same `folds.json`:

```bash
for f in 0 1 2 3 4; do
  python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py \
    --fold-index "$f" \
    --train-producers \
    --epochs 5
done
```

Blend producer artifacts, compute per-candidate AUC, OOF correlations, grid-search
non-negative blend weights, and write the final consumer submission:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py \
  --consume-ensemble
```

Submit the consumer output and log the score when Kaggle exposes one:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py \
  --consume-ensemble \
  --submit \
  --log-memory
```

Consumer outputs:

- `producer_artifacts/candidate_*/fold_*/oof.csv`
- `producer_artifacts/candidate_*/fold_*/test_probs.csv`
- `producer_artifacts/candidate_*/fold_*/producer_manifest.json`
- `ensembles/final_blend/blend_report.json`
- `ensembles/final_blend/pipeline_manifest.json`
- `ensembles/final_blend/submission.csv`
- `ensemble_submission_receipt.json`

## Legacy Single-Model Submission

After single-model training, write a submission receipt without submitting:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --make-submission
```

Submit and log score when Kaggle exposes a public score:

```bash
python experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py --make-submission --submit --log-memory
```

## Generated Candidates

- DINOv3-B/16, partial finetune last 2 blocks
- DINOv3-B/16, partial finetune last 4 blocks
- DINOv2-Base, head-only finetune
- Swin-Base, full finetune with focal loss

The helper enables class weights for all configs because `multiple_diseases` is
rare. Keep the same `folds.json` when comparing candidates.
