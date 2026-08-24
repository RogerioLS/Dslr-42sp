---
title: "[DSLR-11] Inference Engine: Predição Multiclasse e houses.csv"
milestone: "03. Logistic Regression Engine & Math"
labels: ["area: model", "type: implementation", "type: defense"]
---

## 🎯 Objetivo Didático
Carregar os pesos treinados, classificar novos alunos do `dataset_test.csv` e validar a meta de acurácia $\ge 98\%$.

## 📚 Regra de Decisão
Para cada aluno $x$, calculamos a probabilidade de pertencer a cada casa:
$$\hat{y} = \arg\max_{c \in \{\text{Gryffindor}, \text{Hufflepuff}, \text{Ravenclaw}, \text{Slytherin}\}} h_{\theta_c}(x)$$

## 📝 Tarefas Técnicas
- [ ] Criar `logreg_predict.py` na raiz recebendo `dataset_test.csv` e `weights.json`.
- [ ] Normalizar o conjunto de teste usando $\mu$ e $\sigma$ aprendidos no treino.
- [ ] Gerar `houses.csv` no formato estrito do subject:
  ```csv
  Index,Hogwarts House
  0,Gryffindor
  1,Hufflepuff
  ...
  ```
- [ ] Validar acurácia com `make evaluate` ou Scikit-Learn accuracy_score.

## 🧪 Critérios de Aceite
- Acurácia comprovada $\ge 98.0\%$ no conjunto de teste.
