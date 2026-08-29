# Changelog

All notable changes to **42 DSLR (Data Science × Logistic Regression)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Data pipeline module (`src/analytics/loader.py`) for CSV ingestion and NaN handling `[DSLR-01]`.
- Unit test suite for data loading and feature extraction (`tests/unit/test_loader.py`).
- Modular packages scaffolding (`src/analytics/`, `src/preprocessing/`, `src/model/`, `src/visualization/`).
- Automated AST Norm & Anti-Cheating checker (`scripts/norm_check.py`).
- 98% accuracy threshold evaluator (`scripts/evaluate_accuracy.py`).
- Incremental PR summary reporter (`scripts/generate_summary.py`).
- Dynamic PR renamer and automated checklist updater (`scripts/rename_pr.py`, `scripts/update_pr_checklist.py`).
- Interactive ANSI Makefile Command Center with `make check`, `make audit`, `make summary`.
- Dual test suite architecture (`tests/unit/` and `tests/integration/`).
- GitHub CI/CD quality gate enforcement with dynamic PR naming.
- Structured GitHub Issue Templates in YAML (`bug_report.yml`, `task_request.yml`, `math_discussion.yml`).

---

## [1.0.0-rc1] - 2026-08-27

### Added
- Initial project architecture and governance setup.
- Official 42 subject specification and dataset splits (`dataset_train.csv`, `dataset_test.csv`).
- Mathematical derivations documented in `docs/MATHEMATICS.md`.
- Peer evaluation defense walkthrough in `docs/PEER_EVALUATION_GUIDE.md`.
