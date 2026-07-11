# Colab

Open `notebooks/mlestar_kaggle_experiments.ipynb` on the standalone branch.
If data must be downloaded from Kaggle, create a modern API token on Kaggle and
store it in one Colab Secret named `KAGGLE_API_TOKEN`. The notebook maps it only
to the runtime environment and does not print or persist it.

Accept the competition rules in Kaggle before downloading. Store data, models,
OOF predictions and submissions in Drive or private Kaggle datasets, never in
this repository. `mlestar compare` does not submit. The report distinguishes
offline OOF metrics from any accepted public leaderboard result.
