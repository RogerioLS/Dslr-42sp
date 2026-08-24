# 🧙‍♂️ DSLR — Data Science × Logistic Regression (Harry Potter and the Data Scientist)

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Norm 42](https://img.shields.io/badge/norm-42%20compliant-success.svg)
![Target Accuracy](https://img.shields.io/badge/accuracy-%E2%89%A598%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

*A complete, from-scratch Machine Learning classification pipeline recreating Hogwarts' Sorting Hat.*

</div>

---

## 📖 Overview

**DSLR** is a 42 specialization project designed to explore Data Science and Machine Learning fundamentals from first mathematical principles.
The mission is to sort incoming Hogwarts students into one of four houses (**Gryffindor**, **Hufflepuff**, **Ravenclaw**, **Slytherin**) using a custom multi-class Logistic Regression (**One-vs-Rest**) trained via Gradient Descent.

---

## 🎯 Mandatory Deliverables

| Deliverable | Command | Description |
|---|---|---|
| **Data Analysis** | `python3 describe.py datasets/dataset_train.csv` | Handcrafted statistical description of numerical features. |
| **Histogram** | `python3 histogram.py datasets/dataset_train.csv` | Visualizes course score distributions across the 4 houses. |
| **Scatter Plot** | `python3 scatter_plot.py datasets/dataset_train.csv` | Identifies pair of correlated/similar features. |
| **Pair Plot** | `python3 pair_plot.py datasets/dataset_train.csv` | Pairwise scatter matrix for feature selection. |
| **Training** | `python3 logreg_train.py datasets/dataset_train.csv` | Trains One-vs-Rest logistic regression & exports `weights.json`. |
| **Prediction** | `python3 logreg_predict.py datasets/dataset_test.csv weights.json` | Generates `houses.csv` with $\ge 98\%$ accuracy. |

---

## 🏗️ Architecture

```text
Dslr_42/
├── .agents/                    # Agent Operating System & 42 compliance rules
├── .github/                    # CI/CD workflows and PR templates
├── .githooks/                  # Pre-commit git hooks
├── datasets/                   # Training and testing datasets
├── docs/                       # Theoretical math & peer evaluation guides
├── scripts/                    # Norm check, evaluators, hook installer
├── src/                        # Modular engine
│   ├── analytics/              # Handcrafted math statistics (zero lib cheating)
│   ├── visualization/          # Plotting routines
│   ├── preprocessing/          # Data cleaning & standardization
│   └── model/                  # Logistic Regression & Gradient Descent
├── tests/                      # Unit test suites
├── Makefile                    # Command center
└── pyproject.toml              # Dependencies & tool configurations
```

---

## 🚀 Quickstart

```bash
# Setup development environment
make install
make pre-commit

# Run Full Quality & Norm Audit
make audit
```

---

## 👤 Author
* **Rogerio Silva** ([@RogerioLS](https://github.com/RogerioLS)) — *42 São Paulo* 🇧🇷
