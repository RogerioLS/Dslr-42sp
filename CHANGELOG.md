# Changelog

All notable changes to **42 DSLR (Data Science × Logistic Regression)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] - 2026-09-01 — 01. Data Exploration & Handcrafted Stats

### ✨ Features & Algorithms
- **[DSLR-04] CLI describe.py: Visual Formatting and Tabular Alignment** ([#4](https://github.com/RogerioLS/Dslr-42sp/issues/4)) by @RogerioLS
  - Implemented standalone `describe.py` CLI in repository root.
  - Handled CLI arguments (`sys.argv`) with formatted error messages on `sys.stderr`.
  - Formatted table headers and numerical rows with dynamic width and 6 decimal places (`{:>15.6f}`).
  - Validated automated CLI execution via `make describe`.
- **[DSLR-03] Quantile Interpolation: Handcrafted Percentiles (25%, 50%, 75%)** ([#3](https://github.com/RogerioLS/Dslr-42sp/issues/3)) by @RogerioLS
  - Implemented continuous linear interpolation using Method 7 (official Pandas/NumPy standard).
  - Covered boundary conditions ($N=1$, identical values, median and quartiles).
- **[DSLR-02] Math from Scratch: Statistical Engine (Count, Mean, Std, Min, Max)** ([#2](https://github.com/RogerioLS/Dslr-42sp/issues/2)) by @RogerioLS
  - Handcrafted descriptive statistical functions in `src/analytics/statistics.py` from first principles.
  - Sample standard deviation computed with Bessel's correction ($N-1$).
  - Verified 100% compliance against 42 Norm and Anti-Cheating rules.
- **[DSLR-01] Data Pipeline: CSV Loading and NaN Handling** ([#1](https://github.com/RogerioLS/Dslr-42sp/issues/1)) by @RogerioLS
  - Created resilient CSV loader in `src/analytics/loader.py`.
  - Isolated numerical Hogwarts courses from metadata columns (`Index`, `Hogwarts House`, etc.).
  - Implemented column-wise NaN filtering for isolated feature calculations.

---

## [1.0.0-rc1] - 2026-08-27

### Added
- Initial project architecture and governance setup.
- Official 42 subject specification and dataset splits (`dataset_train.csv`, `dataset_test.csv`).
- Mathematical derivations documented in `docs/MATHEMATICS.md`.
- Peer evaluation defense walkthrough in `docs/PEER_EVALUATION_GUIDE.md`.
