# Benchmark status

| Key | Competition | Offline metric | Adapter status |
| --- | --- | --- | --- |
| leaf_classification | Leaf Classification | multiclass log loss | implemented |
| plant_pathology_2020 | Plant Pathology 2020 - FGVC7 | mean ROC-AUC | implemented |
| aptos_2019 | APTOS 2019 Blindness Detection | QWK | implemented |
| dog_breed | Dog Breed Identification | multiclass log loss | implemented |
| aerial_cactus | Aerial Cactus Identification | ROC-AUC | implemented |
| dogs_vs_cats | Dogs vs. Cats Redux | log loss | implemented |
| histopathologic_cancer | Histopathologic Cancer Detection | ROC-AUC | implemented |
| global_wheat | Global Wheat Detection | competition AP | not implemented (object detection) |
| ultrasound_nerve | Ultrasound Nerve Segmentation | Dice | not implemented (segmentation) |
| denoising_dirty_documents | Denoising Dirty Documents | RMSE | not implemented (image denoising) |

The runner produces local fixed-fold OOF comparisons. It writes OOF and test
prediction artifacts but does not write a submission file or call the Kaggle
submission API. Any leaderboard submission must be performed separately and
recorded with its own receipt.
