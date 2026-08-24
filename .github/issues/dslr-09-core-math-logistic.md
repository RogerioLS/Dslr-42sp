---
title: "[DSLR-09] Core Math: Função Sigmoide, Log-Loss e Gradiente Analítico"
milestone: "03. Logistic Regression Engine & Math"
labels: ["area: model", "type: math-heavy", "type: implementation"]
---

## 🎯 Objetivo Didático
Implementar a formulação matemática da Regressão Logística em formato matricial/vetorizado.

## 📚 Fórmulas Fundamentais
1. **Hipótese Sigmoide**:
   $$g(z) = \frac{1}{1 + e^{-z}}, \quad h_\theta(X) = g(X\theta)$$
2. **Custo (Binary Cross-Entropy / Log-Loss)**:
   $$J(\theta) = -\frac{1}{m} \left[ y^T \log(h_\theta(X) + \epsilon) + (1 - y)^T \log(1 - h_\theta(X) + \epsilon) \right]$$
3. **Gradiente do Custo**:
   $$\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)$$

## 📝 Tarefas Técnicas
- [ ] Implementar funções vetorizadas com Numpy em `src/model/logistic_regression.py`.
- [ ] Adicionar estabilidade numérica ($\epsilon = 10^{-15}$) para evitar $\log(0) = -\infty$.
- [ ] Adicionar termo de bias (coluna de $1$s na matriz $X$).

## 🧪 Critérios de Aceite
- Testes unitários validando valores da sigmoide e dimensionalidade das matrizes.
