# Changelog

All notable changes to **42 DSLR (Data Science × Logistic Regression)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] - 2026-09-01 — 01. Data Exploration & Handcrafted Stats

### ✨ Features & Algoritmos
- **[DSLR-04] CLI describe.py: Formatação Visual e Alinhamento de Tabela** ([#4](https://github.com/RogerioLS/Dslr-42sp/issues/4)) by @RogerioLS
  - Criar `describe.py` na raiz do repositório.
  - Tratar argumentos da CLI (`sys.argv`) com mensagens de erro claras se o arquivo não existir.
  - Formatar o cabeçalho e as linhas com `{:>15.6f}` para alinhamento uniforme.
  - Testar com `make describe`.
- **[DSLR-03] Quantile Interpolation: Cálculo de Percentis (25%, 50%, 75%)** ([#3](https://github.com/RogerioLS/Dslr-42sp/issues/3)) by @RogerioLS
  - Implementar algoritmo de ordenação ou usar `sorted()` em lista sem NaNs.
  - Implementar a fórmula de interpolação linear (Método 7 - padrão Pandas).
  - Cobrir casos de borda ($N=1$, mediana e quartis).
- **[DSLR-02] Math from Scratch: Motor Estatístico (Count, Mean, Std, Min, Max)** ([#2](https://github.com/RogerioLS/Dslr-42sp/issues/2)) by @RogerioLS
  - Implementar funções em `src/analytics/statistics.py` a partir de primeiros princípios matemáticos.
  - Desvio padrão amostral com Correção de Bessel ($N-1$).
  - Validar que `make norm` aprova a implementação sem erros.
- **[DSLR-01] Data Pipeline: Carregamento do CSV e Tratamento de NaNs** ([#1](https://github.com/RogerioLS/Dslr-42sp/issues/1)) by @RogerioLS
  - Criar módulo `src/analytics/loader.py` para carregar o CSV.
  - Separar colunas numéricas de colunas categóricas/metadados (`Index`, `Hogwarts House`, etc.).
  - Implementar filtro para desconsiderar NaNs no cálculo de cada coluna individual.

---

## [1.0.0-rc1] - 2026-08-27

### Added
- Initial project architecture and governance setup.
- Official 42 subject specification and dataset splits (`dataset_train.csv`, `dataset_test.csv`).
- Mathematical derivations documented in `docs/MATHEMATICS.md`.
- Peer evaluation defense walkthrough in `docs/PEER_EVALUATION_GUIDE.md`.
