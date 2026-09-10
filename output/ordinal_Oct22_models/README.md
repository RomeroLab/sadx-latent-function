# Ordinal Oct22 Models

This directory contains model configurations, predictions, and probability outputs for ordinal regression models trained on the Oct22 dataset. The final model selected for single mutant recommendations was **6eeae50e**.

## Output Files

Each model produces some or all of the following files:

- `{uuid}.yml` — model configuration
- `{uuid}.probs.txt` — predicted class probabilities
- `{uuid}.preds.txt` — predicted class labels
- `{uuid}.losses.png` — training/validation loss curves (PyTorch models only)
- `{uuid}.ckpt` — saved model checkpoint (PyTorch models, if requested)

## Definitions

### Dataset Splits

| Name | Description |
|------|-------------|
| train | 85% of the full dataset, used for training |
| test | 15% of the full dataset, held out for evaluation |
| train_cv1...train_cv5 | Subsets of train (~70% of total dataset), one per CV fold |
| val_cv1...val_cv5 | Complementary subsets of train (~15% of total dataset), one per CV fold. For any fold i, train = train_cvi + val_cvi |
| all | The entire dataset (train + test combined) |

### Config Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| target | multiclass | Ordinal classification into 4 categories: N=0, L=1, P=2, H=3 using CORAL loss |
| target | binary | Binary classification: inactive (N=0) vs active (L, P, H all = 1) using binary cross entropy |
| encoding | one-hot | Each amino acid at each position is represented as a 20-dimensional binary vector |
| dca | true/false | Whether a bmDCA co-evolutionary density score is appended as an additional input feature |
| multilibrary | true/false | When true, each sample carries its library index (0, 1, or 2) and the CORAL output layer uses library-specific bias terms. When false, all samples are treated as belonging to a single library |
| intercept | true/false | Whether the model includes an intercept/bias term (sklearn models only) |
| lambda_h | float | L2 regularization on output layer weights. Recorded in config but **not used** by LightningMLP |
| lambda_dca | float | L2 regularization on the DCA feature weight. Recorded in config but **not used** by LightningMLP |
| weight_decay | float | L2 regularization applied through the Adam optimizer |

### Activity Categories

| Category | Code | Description |
|----------|------|-------------|
| N (Dead) | 0 | No activity |
| L (Low) | 1 | Low activity |
| P (Parent) | 2 | Parent-like activity |
| H (High) | 3 | Higher than parent activity |

Activity labels were assigned relative to each parent library, meaning the thresholds separating N, L, P, and H differ across libraries.

## Cross Validation

All models draw from the same set of CV splits defined in the dataset. The full dataset is split into train (85%) and test (15%). The train split is further divided into 5 CV folds, each consisting of a train_cv subset (~70% of total) and a val_cv subset (~15% of total).

Sklearn and PyTorch models use these splits differently but draw from the **same predefined folds**. The dataset object implements sklearn's CV splitter interface (`split()` and `get_n_splits()`), so when passed as `cv=ds` to `LogisticRegressionCV`/`RidgeClassifierCV`, sklearn internally iterates over the same 5 folds to select the best regularization hyperparameter, then refits on the full train split. PyTorch models only use a single fold (val_cv1) for monitoring validation loss during training, with no automatic hyperparameter selection across folds. Additionally, the final PyTorch model (6eeae50e) trains on `all` (the entire dataset including the test set), while sklearn models always train on the `train` split only.

## MLPLightning (CORAL Neural Network)

All MLPLightning models use the `LightningMLP` code path, which wraps a `BaseCoralModule`. Note that λ_h and λ_dca are recorded in the config but **not used** in the `LightningMLP` training loop — regularization is provided only by dropout and optimizer weight decay. Hidden layer sizes (100, 50), dropout (0.2), and batch size (32) are constant across all runs.

| Label           | UUID | Train Set | Val Set | Test Set | Target | Epochs | Learning Rate | Weight Decay | DCA | Multilibrary |
|-----------------|------|-----------|---------|----------|--------|--------|---------------|--------------|-----|--------------|
| MLP-A           | 35804fd8 | train | val_cv1 | test | multiclass | 50 | 0.01 | 1e-4 | no | no |
| MLP-B           | 140d9f93 | train | val_cv1 | test | multiclass | 50 | 0.01 | 1e-4 | yes | no |
| MLP          | 5a7103a1 | train | val_cv1 | test | multiclass | 50 | 0.01 | 1e-4 | yes | yes |
| MLP (epochs 1)  | 32cc250e | all | val_cv1 | test | multiclass | 1 | 3e-4 | 1e-6 | yes | yes |
| MLP (epochs 5)  | 7b976172 | all | val_cv1 | test | multiclass | 5 | 3e-4 | 1e-6 | yes | yes |
| MLP (epochs 25) | 92715f9c | all | val_cv1 | test | multiclass | 25 | 3e-4 | 1e-6 | yes | yes |
| **MLP-final**         | **6eeae50e** | **all** | **val_cv1** | **test** | **multiclass** | **400** | **3e-4** | **1e-6** | **yes** | **yes** |

All MLPLightning runs produce loss curves (`.losses.png`), predictions (`.preds.txt`), and probabilities (`.probs.txt`).

## Sklearn Models

Sklearn models are run via `model_sklearn_Oct22.py`. They do not produce loss curves.

| Label | UUID | Model | Target | Train Set | Internal CV (same 5 folds as above) | Test Set | Intercept | DCA | Multilibrary |
|-------|------|-------|--------|-----------|--------------------------------------|----------|-----------|-----|--------------|
| Linear (no intercept) | 6d97d935 | LogisticRegression | multiclass | train | 5-fold | test | no | no | no |
| Linear | 9a10c5e1 | LogisticRegression | multiclass | train | 5-fold | test | yes | no | no |
| Linear-bin (no intercept) | ae68d863 | LogisticRegression | binary | train | 5-fold | test | no | no | no |
| Linear-bin | cfeb8a05 | LogisticRegression | binary | train | 5-fold | test | yes | no | no |
| Ridge (no intercept) | 07df498c | RidgeClassifier | multiclass | train | 5-fold | test | no | no | no |
| Ridge | 890b899b | RidgeClassifier | multiclass | train | 5-fold | test | yes | no | no |
| Ridge-bin (no intercept) | 049830f8 | RidgeClassifier | binary | train | 5-fold | test | no | no | no |
| Ridge-bin | 4450b384 | RidgeClassifier | binary | train | 5-fold | test | yes | no | no |

All sklearn models are trained on the train split and evaluated on the test split. Sklearn uses all 5 CV folds internally for hyperparameter selection (see Cross Validation section above). The sklearn configs do not contain `dataset_params` as the train/test split is hardcoded in the script. They produce predictions (`.preds.txt`) and probabilities (`.probs.txt`) but no loss curves.
