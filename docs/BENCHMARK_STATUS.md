# Benchmark status

The project supports the following historical Kaggle task contracts:

| Key | Competition | Offline metric | Submission status |
| --- | --- | --- | --- |
| plant_pathology_2020 | Plant Pathology 2020 - FGVC7 | mean ROC-AUC | expected closed |
| aptos_2019 | APTOS 2019 Blindness Detection | QWK | expected closed |
| dog_breed | Dog Breed Identification | multiclass log loss | expected closed |
| global_wheat | Global Wheat Detection | competition AP | expected closed |
| ultrasound_nerve | Ultrasound Nerve Segmentation | Dice | expected closed |
| leaf_classification | Leaf Classification | multiclass log loss | expected closed |
| aerial_cactus | Aerial Cactus Identification | ROC-AUC | expected closed |
| dogs_vs_cats | Dogs vs. Cats Redux | log loss | expected closed |
| histopathologic_cancer | Histopathologic Cancer Detection | ROC-AUC | expected closed |
| denoising_dirty_documents | Denoising Dirty Documents | RMSE | expected closed |

The runner first produces local fixed-fold OOF comparisons and validates a local
submission against the sample submission. If a user explicitly requests a
submission, it records the exact Kaggle acceptance or rejection response.
