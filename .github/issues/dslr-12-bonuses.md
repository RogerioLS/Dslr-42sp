---
title: "[DSLR-12] Bonus: Otimizadores (SGD / Mini-Batch GD) & Estatísticas Extras"
milestone: "04. Defense Readiness & Bonuses"
labels: ["area: model", "type: bonus", "type: implementation"]
---

## 🎯 Objetivo Didático
Explorar variações do algoritmo de otimização e enriquecer a análise estatística.

## 📚 Conceitos dos Bônus
- **SGD (Stochastic Gradient Descent)**: Atualiza $\theta$ a cada amostra individual ($m=1$). Muito rápido, mas com gradiente ruidoso.
- **Mini-Batch Gradient Descent**: Atualiza $\theta$ em lotes pequenos (ex: batch de 32 ou 64). Equilibra velocidade e estabilidade vetorial.
- **Métricas Extras no describe.py**: Skewness (assimetria), Kurtosis (curtose), IQR, Variância e contagem de NaNs.

## 📝 Tarefas Técnicas
- [ ] Adicionar flag `--method batch/sgd/minibatch` no `logreg_train.py`.
- [ ] Implementar estatísticas adicionais no `src/analytics/statistics.py`.

## 🧪 Critérios de Aceite
- Comparação de tempo de execução e curvas de convergência documentadas.
