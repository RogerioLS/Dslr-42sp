---
title: "[DSLR-04] CLI describe.py: Formatação Visual e Alinhamento de Tabela"
milestone: "01. Data Exploration & Handcrafted Stats"
labels: ["area: stats", "type: implementation"]
---

## 🎯 Objetivo Didático
Construir a interface de linha de comando oficial da primeira parte do subject e formatar o output no terminal de maneira profissional.

## 📚 Conceito para Estudo em Dupla
O subject exige que `describe.py` receba um arquivo CSV como parâmetro e exiba a tabela de todas as features numéricas com 6 casas decimais, alinhadas por colunas.

## 📝 Tarefas Técnicas
- [ ] Criar `describe.py` na raiz do repositório.
- [ ] Tratar argumentos da CLI (`sys.argv`) com mensagens de erro claras se o arquivo não existir.
- [ ] Formatar o cabeçalho e as linhas com `{:>15.6f}` para alinhamento uniforme.
- [ ] Testar com `make describe`.

## 🧪 Critérios de Aceite
- Execução: `python3 describe.py datasets/dataset_train.csv`.
- Saída visual idêntica à página 6 do PDF do subject.
