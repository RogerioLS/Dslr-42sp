---
title: "[DSLR-10] Training Engine: One-vs-Rest (OvR) e Batch Gradient Descent"
milestone: "03. Logistic Regression Engine & Math"
labels: ["area: model", "type: implementation", "type: math-heavy"]
---

## 🎯 Objetivo Didático
Construir o pipeline de treinamento supervisionado multiclasse para as 4 casas de Hogwarts.

## 📚 Conceito para Estudo em Dupla
Como a Regressão Logística é naturalmente binária ($0$ ou $1$), o método **One-vs-Rest (OvR)** treina 4 modelos separados:
1. Gryffindor ($1$) vs Outras ($0$)
2. Hufflepuff ($1$) vs Outras ($0$)
3. Ravenclaw ($1$) vs Outras ($0$)
4. Slytherin ($1$) vs Outras ($0$)

## 📝 Tarefas Técnicas
- [ ] Criar executável `logreg_train.py` na raiz.
- [ ] Implementar loop de Batch Gradient Descent com hiperparâmetros ajustáveis (learning rate $\alpha$, epochs).
- [ ] Salvar os pesos $\theta$ das 4 casas e parâmetros do scaler em `weights.json`.
- [ ] Exibir a evolução da perda ($J(\theta)$) a cada época para checar convergência.

## 🧪 Critérios de Aceite
- Execução: `python3 logreg_train.py datasets/dataset_train.csv`.
- Geração bem-sucedida do arquivo de pesos.
