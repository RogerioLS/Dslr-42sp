---
title: "[DSLR-01] Data Pipeline: Carregamento do CSV e Tratamento de NaNs"
milestone: "01. Data Exploration & Handcrafted Stats"
labels: ["area: stats", "type: implementation", "type: pedagogical"]
---

## 🎯 Objetivo Didático
Aprender como estruturar a ingestão de dados tabulares e o manuseio de valores ausentes (*missing data/NaNs*) sem depender de métodos prontos de imputação.

## 📚 Conceito para Estudo em Dupla
Em Ciência de Dados, dados brutos quase nunca estão limpos. Alunos de Hogwarts podem ter faltado em exames (valores vazios no CSV). Precisamos identificar colunas puramente numéricas e ignorar ou tratar valores ausentes durante cálculos analíticos sem quebrar o algoritmo.

## 📝 Tarefas Técnicas
- [ ] Criar módulo `src/analytics/loader.py` ou parser no pandas/numpy para carregar o CSV.
- [ ] Separar colunas numéricas de colunas categóricas/metadados (`Index`, `Hogwarts House`, `First Name`, `Last Name`, `Birthday`, `Best Hand`).
- [ ] Implementar filtro para desconsiderar NaNs no cálculo de cada coluna individual.

## 🧪 Critérios de Aceite
- Extração de matriz ou arrays de features numéricas funcionais.
- Preservação da integridade posicional dos dados.
