---
title: "[DSLR-06] Bivariate Analysis: Identificação de Features Correlacionadas (Scatter Plot)"
milestone: "02. Data Visualization & Feature Analysis"
labels: ["area: visualization", "type: implementation", "type: pedagogical"]
---

## 🎯 Objetivo Didático
Analisar a correlação bivariada entre pares de matérias para identificar redundâncias no dataset.

## 📚 Pergunta do Subject
*What are the two features that are similar?*

## 📚 Conceito para Estudo em Dupla
Se duas matérias possuem uma relação linear quase perfeita (correlação de Pearson $\approx 1.0$), incluir ambas no modelo de Regressão Logística adiciona multicolinearidade sem trazer informação nova.

## 📝 Tarefas Técnicas
- [ ] Criar `scatter_plot.py` na raiz.
- [ ] Calcular matriz de correlação de Pearson para guiar a busca visual.
- [ ] Plotar o gráfico de dispersão com os pontos das 4 casas coloridos.
- [ ] Documentar o par de features correlacionadas em `docs/DATA_VISUALIZATION.md`.

## 🧪 Critérios de Aceite
- Gráfico de dispersão claro mostrando a correlação evidente entre as duas matérias.
