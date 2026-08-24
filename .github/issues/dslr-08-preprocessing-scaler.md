---
title: "[DSLR-08] Preprocessing: Normalização Z-Score (StandardScaler)"
milestone: "03. Logistic Regression Engine & Math"
labels: ["area: model", "type: math-heavy", "type: implementation"]
---

## 🎯 Objetivo Didático
Entender a importância do escalonamento de features para o algoritmo de Gradiente Descendente.

## 📚 Conceito para Estudo em Dupla
Se uma matéria varia de $0$ a $10$ e outra varia de $-1000$ a $+1000$, as curvas de nível da função de custo $J(\theta)$ tornam-se elipses extremamente estreitas. O gradiente oscilará em zigue-zague e demorará muito para convergir.
Com **StandardScaler (Z-score)**:
$$z = \frac{x - \mu}{\sigma}$$
Todas as features passam a ter média $0$ e desvio padrão $1$, tornando o espaço de perda esférico e a descida do gradiente estável.

## 📝 Tarefas Técnicas
- [ ] Criar classe `StandardScaler` artesanal em `src/preprocessing/scaler.py`.
- [ ] Implementar métodos `fit(X)`, `transform(X)` e `fit_transform(X)`.
- [ ] Salvar as médias $\mu$ e desvios $\sigma$ do treino para aplicar identicamente no teste (sem data leakage!).

## 🧪 Critérios de Aceite
- Média pós-transformação $\approx 0$ e desvio padrão $\approx 1$ em todas as colunas numéricas.
