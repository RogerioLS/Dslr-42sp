# Copilot & Agent Instructions — 42 DSLR (Data Science x Logistic Regression)

This repository contains the complete implementation, tests, and peer-evaluation defense documentation for **DSLR (Data Science × Logistic Regression — Harry Potter and the Data Scientist)** at École 42 São Paulo.

---

## 🎯 Project Scope & Architecture

1. **Data Analysis (`describe.py` / `src/analytics/`)**:
   - Pure mathematical implementation of statistical aggregates (`Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max`).
   - ⚠️ **Strict 42 Anti-Cheating**: Strictly prohibited to use built-in functions like `df.describe()`, `df.mean()`, `df.std()`, `np.mean()`, `np.std()`, `np.percentile()`.

2. **Data Visualization (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)**:
   - `histogram.py`: Identifies the course with homogeneous distribution across houses.
   - `scatter_plot.py`: Identifies the two collinear/similar features.
   - `pair_plot.py`: Generates the full feature matrix to justify feature selection.

3. **Multi-class Logistic Regression (`logreg_train.py`, `logreg_predict.py`, `src/model/`, `src/preprocessing/`)**:
   - Handcrafted **One-vs-Rest (OvR)** binary classifier pipeline.
   - Batch Gradient Descent minimizing Binary Cross-Entropy Loss ($J(\theta)$).
   - Sigmoid activation function ($g(z) = \frac{1}{1 + e^{-z}}$).
   - Feature standardization (Z-score StandardScaler: $z = \frac{x - \mu}{\sigma}$).
   - Generates `houses.csv` achieving $\ge 98.0\%$ accuracy on `dataset_test.csv`.

---

## 🛡️ 42 Principles & Standards

Always favor:
- Minimal, clean, and mathematically sound code.
- Explicit function and class docstrings (`__doc__`) on every function, class, and method.
- Main entrypoint guards (`if __name__ == "__main__":`) on all executable scripts.
- Maximum line length of 100 characters adhering to Black and Flake8 standards.
- Vectorized NumPy operations over slow manual loops for linear algebra.
- Verification via Makefile (`make audit`, `make norm`, `make test`, `make evaluate`).

Avoid:
- Using banned third-party functions for statistical calculations.
- Hardcoded local paths (always accept CLI arguments with sensible defaults).
- Swallowing exceptions silently without descriptive error feedback.
- Committing temporary caches (`__pycache__`, `.pytest_cache`, `.DS_Store`).

---

## 🧪 Definition of Done

A task is complete only when:
1. Syntax compiles cleanly (`make compile`).
2. 42 Norm & AST Anti-Cheating Auditor passes with 0 errors (`make norm`).
3. Unit tests pass (`make test`).
4. Output formatting matches the 42 subject specifications.
5. Accuracy on `dataset_test.csv` meets or exceeds $98.0\%$ (`make evaluate`).
