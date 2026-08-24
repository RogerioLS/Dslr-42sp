# AGENTS.md — 42 DSLR (Data Science x Logistic Regression) Agent Operating System

**Project:** 42 DSLR — Data Science × Logistic Regression
**Curriculum:** 42 São Paulo / 42 Network Specialization
**Python:** >= 3.10
**Purpose:** Master operating protocol for AI coding agents working in this repository
**Version:** 1.0

---

## 0. Why This File Exists

This file is the repository-level operating protocol for AI agents.
It is the control layer that defines:
- The exact delivery contract expected by 42 peer-evaluators;
- Mathematical boundaries and strict no-cheating rules (zero built-in statistical functions);
- The required files, formats, and command line interfaces;
- Guidelines for peer-defense readiness.

---

## 1. Institutional Identity & Philosophy

You are operating as a strict 42 AI Coding Assistant.
- **Source of truth:** The subject PDF (`subject/en.subject.pdf`) and repository files.
- **Anti-Cheating:** It is strictly forbidden to use library functions that do the math for you (e.g., `df.describe()`, `np.mean()`, `df.std()`, `np.percentile()`). All statistical aggregates must be computed from raw mathematical definitions.
- **Accuracy Contract:** The classifier evaluated on `dataset_test.csv` must achieve $\ge 98\%$ accuracy.
- **Code Quality:** All functions/classes require complete docstrings, entrypoint guards, and PEP8/42 line length constraints ($\le 100$ chars).

---

## 2. Deliverable Requirements

| File | Type | Purpose | Mandatory Constraints |
|---|---|---|---|
| `describe.py` | Executable CLI | Display numerical features stats | No `describe()`, `mean()`, `std()`, etc. |
| `histogram.py` | Executable CLI | Plot course score distributions | Answers: Which course is homogeneous across houses? |
| `scatter_plot.py` | Executable CLI | Plot bivariate scatter | Answers: What are the two most similar features? |
| `pair_plot.py` | Executable CLI | Plot scatter plot matrix | Answers: Which features are selected for training? |
| `logreg_train.py` | Executable CLI | Train Logistic Regression | Custom Gradient Descent & One-vs-Rest model |
| `logreg_predict.py`| Executable CLI | Generate `houses.csv` | Exactly: `Index,Hogwarts House` |

---

## 3. Directory Architecture

- `src/analytics/`: Handcrafted mathematical statistics algorithms.
- `src/visualization/`: Data plotting routines and styling.
- `src/preprocessing/`: Data cleaning, missing value handling, feature scaling.
- `src/model/`: Multi-class One-vs-Rest Logistic Regression & Gradient Descent.
- `tests/`: Automated unit test suites.
- `scripts/`: Norm checks, Git hooks, evaluation tools.
- `docs/`: Mathematical derivations, visualization analyses, defense notes.
