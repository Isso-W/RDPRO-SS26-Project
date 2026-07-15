# Colab

Open either notebook from `experiments/mlestar_kaggle_benchmarks/notebooks/` on
the Jiaozi `main` branch. Each notebook clones `https://github.com/Isso-W/Jiaozi`
at `main`, then changes into `experiments/mlestar_kaggle_benchmarks` before
installing the experiment dependencies.

If data must be downloaded from Kaggle, create a modern API token on Kaggle and
store it in exactly one Colab Secret named `KAGGLE_API_TOKEN`. The notebooks
copy the secret only into the current process environment. They do not display
it, write a credential file, or include it in saved notebook output.

Accept the competition rules in Kaggle before downloading. Store data, models,
OOF predictions and submissions in Drive or private Kaggle datasets, never in
this repository. `mlestar compare` does not submit. The report distinguishes
offline OOF metrics from any separately recorded public leaderboard result.
