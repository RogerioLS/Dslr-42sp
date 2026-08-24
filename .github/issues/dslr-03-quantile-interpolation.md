---
title: "[DSLR-03] Quantile Interpolation: Cálculo de Percentis (25%, 50%, 75%)"
milestone: "01. Data Exploration & Handcrafted Stats"
labels: ["area: stats", "type: math-heavy", "type: implementation"]
---

## 🎯 Objetivo Didático
Entender a matemática da ordenação e da interpolação linear para determinar quartis e mediana.

## 📚 Conceito para Estudo em Dupla
- **25% (Q1 - Primeiro Quartil)**: Valor abaixo do qual estão 25% dos dados.
- **50% (Q2 - Mediana)**: Ponto central que divide o conjunto em duas metades iguais.
- **75% (Q3 - Terceiro Quartil)**: Valor abaixo do qual estão 75% dos dados.
- **Fórmula de Interpolação Linear (Método 7 - padrão Pandas/Numpy)**:
  $$\text{index} = (N - 1) \cdot p$$
  $$\text{Percentil}(p) = X[\lfloor \text{index} \rfloor] + (\text{index} - \lfloor \text{index} \rfloor) \cdot (X[\lceil \text{index} \rceil] - X[\lfloor \text{index} \rfloor])$$

## 📝 Tarefas Técnicas
- [ ] Implementar algoritmo de ordenação ou usar `sorted()` em lista sem NaNs.
- [ ] Implementar a fórmula de interpolação linear.
- [ ] Cobrir casos de borda ($N=1$, valores idênticos, lista vazia).

## 🧪 Critérios de Aceite
- Testes unitários em `tests/test_statistics.py` validando valores de $25\%$, $50\%$ e $75\%$ contra o output oficial.
