---
title: "[DSLR-02] Math from Scratch: Motor Estatístico (Count, Mean, Std, Min, Max)"
milestone: "01. Data Exploration & Handcrafted Stats"
labels: ["area: stats", "type: math-heavy", "type: implementation"]
---

## 🎯 Objetivo Didático
Implementar as operações estatísticas fundamentais do zero em Python puro, respeitando a regra estrita de **No-Cheating** da 42.

## 📚 Conceito para Estudo em Dupla
- **Count ($N$)**: Total de valores não nulos.
- **Mean ($\mu$)**: $\mu = \frac{1}{N} \sum x_i$. Representa o ponto de equilíbrio dos dados.
- **Sample Variance ($\sigma^2$)**: $\sigma^2 = \frac{1}{N-1} \sum (x_i - \mu)^2$. Usa-se $N-1$ (correção de Bessel) porque estamos trabalhando com uma amostra.
- **Standard Deviation ($\sigma$)**: $\sigma = \sqrt{\sigma^2}$. Mede o grau de dispersão em torno da média.
- **Min / Max**: Menor e maior valor finito do conjunto.

## ⚠️ Regra 42 Anti-Cheating
É expressamente proibido usar `df.describe()`, `df.mean()`, `df.std()`, `np.mean()`, `np.std()`, etc.

## 📝 Tarefas Técnicas
- [ ] Implementar funções em `src/analytics/statistics.py`.
- [ ] Garantir complexidade de tempo linear $O(N)$ para média e variância.
- [ ] Validar que `make norm` aprova a implementação sem erros.

## 🧪 Critérios de Aceite
- Resultados idênticos aos do Pandas/Numpy com tolerância de $10^{-6}$.
