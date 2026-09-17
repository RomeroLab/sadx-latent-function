# Cross-validation and balanced accuracy

Upload this entire folder at the repository root. It contains all analysis and
training code needed for the table; no imports from the previous audit scripts
or PyTorch Lightning are required. Successful execution prints only the table.

## Run

From the repository root, using Python 3.10 or newer:

```bash
python -m pip install -r cv_balanced_accuracy/requirements.txt
python cv_balanced_accuracy/run.py
```

The default run trains 75 neural networks: five ablations, five held-out folds,
and three initialization seeds. It runs on CPU with one thread, so allow time
for training. Completed fits are cached and reused on subsequent runs with the
same data, configuration, code, and environment. Progress is written to
`cv_balanced_accuracy/cache/<run-id>/run.log`, not printed. The cache also stores
the run manifest, fold assignments, and row-level out-of-fold probabilities.
The cache is ignored by Git and need not be uploaded.

For a quick installation check (not scientific results):

```bash
python cv_balanced_accuracy/run.py --smoke
```

To print the table from an existing prediction file without retraining:

```bash
python cv_balanced_accuracy/run.py --predictions output/coral_cv_6ee_recipe_20260912/predictions.csv
```

That existing prediction file is optional and is not needed for a fresh run.
The command can be invoked from any working directory. All relative input and
cache paths resolve against the repository root derived from this file's
location. No machine-specific absolute paths are embedded in the code.

## Existing repository inputs

- `../output/ordinal_Oct22_sequences_with_dca_score.csv`: labeled sequences,
  parent-library identities, and precomputed DCA scores.
- `../output/ordinal_Oct22_models/saved_models/6eeae50e.yml`: final design-model
  training settings (400 epochs, Adam learning rate 0.0003, weight decay
  0.000001, batch size 32).

These paths are shown relative to this folder. `--data` and `--config` can
override them, using paths relative to the repository root.

The model trained on all data is the source of the architecture and training
recipe. Its checkpoint is deliberately not used to initialize CV fits, since
it has already seen their held-out sequences. Each fit starts from fresh
random weights. Hidden layers (100, 50), ReLU activations, dropout 0.2, and the
CORAL loss follow `../scripts/model_pytorch_Oct22.py`. The first layer evaluates
the one-hot linear transformation using an equivalent sparse embedding sum.
The DCA score is appended as one input; library identity selects CORAL thresholds.

| Model | Sequence | DCA | Library-specific thresholds |
|---|---|---|---|
| Full | Yes | Yes | Yes |
| No DCA | Yes | No | Yes |
| No library bias | Yes | Yes | No |
| Sequence only | Yes | No | No |
| DCA only | No | Yes | Yes |

The library-prior baseline estimates each library's class frequencies from the
training fold only. Always Low assigns probability one to L.

## Evaluation

Five-fold `StratifiedGroupKFold` uses joint parent/category strata, grouping
identical sequences so none occur in both training and held-out data. The split
seed is 20260912. Rare strata cannot all appear in every fold. The same folds
are used across all ablations and training seeds (100, 101, 102). These are
three initializations on one fixed partition, not three different CV splits.

For each training seed, held-out predictions are pooled across all five folds.
The predicted class is the argmax of the four bin probabilities. Balanced
accuracy is `(recall_N + recall_L + recall_P + recall_H) / 4`. We calculate it
within each parent and across all sequences, then average each score across
the three training seeds. "All" pools sequences; it is not the mean of the
three parent scores. The metric weights each class uniformly. No threshold or
hyperparameter is tuned against held-out predictions.

Expected table from the existing 2026-09-12 CV predictions:

```text
                          1VH    2L   3VRL    All
Full model              0.272  0.265  0.256  0.322
No DCA                  0.263  0.272  0.281  0.317
No library bias         0.274  0.264  0.263  0.269
Sequence only           0.267  0.271  0.270  0.282
DCA only                0.250  0.250  0.243  0.309
Library-prior baseline  0.250  0.250  0.250  0.293
Always-Low baseline     0.250  0.250  0.250  0.250
```

Fresh training may differ numerically across PyTorch versions or platforms.
The earlier run used NumPy 1.26.4, pandas 2.1.0, scikit-learn 1.7.0, and
PyTorch 2.14.0 on CPU. The manifest records the actual versions for each run.

These are retrospective ablation results for the final model recipe. The
aggregate score may benefit from differences among libraries; the parent
columns are necessary to assess sequence discrimination within each library.
