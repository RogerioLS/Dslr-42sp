# Peer Evaluation Defense Guide for 42 DSLR

## Checklist for Evaluation Day

### 1. Data Analysis (`describe.py`)
- [ ] Run `python3 describe.py datasets/dataset_train.csv`
- [ ] Compare output against `pandas.DataFrame.describe()` to prove exact decimal matching.
- [ ] Show the evaluator `src/analytics/statistics.py` to prove zero forbidden library calls.

### 2. Data Visualization
- [ ] Run `python3 histogram.py datasets/dataset_train.csv` and state the homogeneous course.
- [ ] Run `python3 scatter_plot.py datasets/dataset_train.csv` and show the correlated features.
- [ ] Run `python3 pair_plot.py datasets/dataset_train.csv` and justify the selected feature subset.

### 3. Model Training (`logreg_train.py`)
- [ ] Run `python3 logreg_train.py datasets/dataset_train.csv`
- [ ] Demonstrate convergence of cost function $J(\theta)$ over epochs.
- [ ] Verify that `weights.json` / `weights.csv` is produced.

### 4. Prediction & Accuracy (`logreg_predict.py`)
- [ ] Run `python3 logreg_predict.py datasets/dataset_test.csv weights.json` (or `make predict`).
- [ ] Verify execution completes without errors and outputs `houses.csv`.
- [ ] Check format of generated `houses.csv`:
  ```csv
  Index,Hogwarts House
  0,Hufflepuff
  1,Ravenclaw
  ...
  ```
- [ ] Confirm exact 401 lines (header + 400 test students).
- [ ] Validate accuracy against ground truth:
  ```bash
  python3 scripts/evaluate_accuracy.py houses.csv /path/to/dataset_truth.csv
  # or using Makefile:
  make evaluate TRUTH_DATA=/path/to/dataset_truth.csv
  ```
- [ ] Confirm accuracy score meets or exceeds the required **98.00%** threshold.
- [ ] Explain zero data leakage: test set features are scaled using $\mu$ and $\sigma$ learned during training, and missing values impute to $z=0.0$.
