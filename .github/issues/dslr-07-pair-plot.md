---
title: "[DSLR-07] Multivariate Matrix: Pair Plot e Seleção de Features"
milestone: "02. Data Visualization & Feature Analysis"
labels: ["area: visualization", "type: implementation", "type: pedagogical"]
---

## 🎯 Objetivo Didático
Gerar uma matriz de dispersão completa (*pair plot / scatter matrix*) para fundamentar matematicamente quais features serão usadas no treinamento.

## 📚 Pergunta do Subject
*From this visualization, which features are you going to use for your logistic regression?*

## 📚 Conceito para Estudo em Dupla
Um Pair Plot plota todas as combinações de 2 a 2 matérias (gráficos de dispersão fora da diagonal) e a densidade/histograma na diagonal principal. As melhores features são aquelas onde as 4 nuvens de cores estão claramente separadas no espaço.

## 📝 Tarefas Técnicas
- [ ] Criar `pair_plot.py` na raiz utilizando Seaborn/Matplotlib.
- [ ] Documentar a lista final de features selecionadas justificando a exclusão das matérias homogêneas e redundantes.

## 🧪 Critérios de Aceite
- Matriz completa gerada sem cortes de escala.
- Relatório de justificativa em `docs/DATA_VISUALIZATION.md`.
