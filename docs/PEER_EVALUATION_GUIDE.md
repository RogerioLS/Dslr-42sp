# Peer Evaluation Defense Guide & Mock Interview (42 DSLR)

Este guia foi elaborado para garantir nota máxima (100% ou 125% com bônus) na avaliação presencial entre pares da École 42. Ele contém o roteiro de demonstração ao vivo, as perguntas clássicas dos avaliadores e as respostas técnicas esperadas.

---

## 📋 1. Roteiro Prático de Demonstração (Live Walkthrough)

### Etapa 0: Auditoria Anti-Cheating e Testes Automatizados
Antes de começar, mostre ao avaliador que o código é 100% autêntico e cumpre a Norma:
```bash
make audit
```
- **O que acontece**: Executa compilação estrita Python 3.10, auditoria AST que varre o código à procura de funções proibidas do Pandas/NumPy (`describe`, `mean`, `std`, `percentile`, `sklearn`, etc.) e roda os **107 testes automatizados**.

---

### Etapa 1: Análise de Dados (`describe.py`)
Execute o script no dataset de treino:
```bash
python3 describe.py datasets/dataset_train.csv
```
- **Bônus de Estatísticas Extras**:
  ```bash
  python3 describe.py datasets/dataset_train.csv --bonus
  ```
- **O que mostrar ao avaliador**:
  1. Abra `src/analytics/statistics.py` e aponte as funções `compute_mean`, `compute_std`, `compute_percentile`, `compute_variance`, `compute_iqr`, `compute_skewness` e `compute_kurtosis`.
  2. Destaque que o cálculo de desvio padrão utiliza a **correção de Bessel** ($N - 1$) para estimativa amostral não-enviesada.
  3. Destaque que os percentis utilizam **interpolação linear contínua (Método 7)** idêntica ao Pandas e NumPy.

---

### Etapa 2: Visualização de Dados (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)

1. **Histograma**:
   ```bash
   python3 histogram.py datasets/dataset_train.csv
   ```
   - **Pergunta do Avaliador**: *"Qual matéria tem uma distribuição homogênea entre todas as 4 casas de Hogwarts?"*
   - **Resposta**: **Care of Magical Creatures** (Cuidado com Criaturas Mágicas). As curvas de densidade das 4 casas se sobrepõem perfeitamente em formato gaussiano unimodal, provando que essa disciplina não discrimina alunos para as casas.

2. **Scatter Plot**:
   ```bash
   python3 scatter_plot.py datasets/dataset_train.csv
   ```
   - **Pergunta do Avaliador**: *"Quais são as duas matérias mais semelhantes / correlacionadas?"*
   - **Resposta**: **Astronomy** e **Defense Against the Dark Arts**. Elas apresentam correlação linear negativa perfeita ($r = -1.000000$). Saber a nota de uma permite deduzir a outra diretamente.

3. **Pair Plot**:
   ```bash
   python3 pair_plot.py datasets/dataset_train.csv
   ```
   - **Pergunta do Avaliador**: *"Quais features você selecionou para o treinamento e por quê?"*
   - **Resposta**: Descartamos *Care of Magical Creatures* (homogênea, sem poder separador) e uma entre *Astronomy* e *Defense Against the Dark Arts* (colineares, informação redundante). As matérias com separação bimodal nítida (ex: *Herbology*, *Divination*, *Muggle Studies*, *Ancient Runes*) foram priorizadas.

---

### Etapa 3: Treinamento do Modelo (`logreg_train.py`)

Treine o modelo em modo padrão (Batch Gradient Descent):
```bash
python3 logreg_train.py datasets/dataset_train.csv
```
- Mostre a convergência da função de custo $J(\theta)$ (Log-Loss) decrescendo monotonicamente a cada época.
- Verifique que o arquivo `weights.json` foi gerado com os pesos dos 4 classificadores e os parâmetros do scaler ($\mu, \sigma$).

**Demonstração dos Bônus de Otimização**:
```bash
# Mini-Batch Gradient Descent
python3 logreg_train.py datasets/dataset_train.csv --method minibatch --batch-size 64

# Stochastic Gradient Descent (SGD)
python3 logreg_train.py datasets/dataset_train.csv --method sgd
```

---

### Etapa 4: Predição e Acurácia (`logreg_predict.py`)

Gere o arquivo `houses.csv`:
```bash
python3 logreg_predict.py datasets/dataset_test.csv weights.json
```
- **Verificações de formato**:
  1. `head -n 5 houses.csv` ➔ Cabeçalho exato: `Index,Hogwarts House`.
  2. `wc -l houses.csv` ➔ Exatamente 401 linhas (1 cabeçalho + 400 alunos).
- **Validação de Acurácia**:
  ```bash
  python3 scripts/evaluate_accuracy.py houses.csv datasets/dataset_truth.csv
  ```
  *(Ou via `make evaluate`)*. A acurácia deve ser $\ge 98.0\%$ (o modelo atinge tipicamente $\approx 98.5\%-99.0\%$).

---

## 🧠 2. Mock Interview: Perguntas & Respostas Conceituais

### P1: Como funciona a Regressão Logística em problemas Multiclasse se ela é nativamente binária?
> **R:** Usamos a estratégia **One-vs-Rest (OvR)** ou *One-vs-All*. Treinamos 4 modelos binários independentes:
> 1. Grifinória vs (Lufa-Lufa + Corvinal + Sonserina)
> 2. Lufa-Lufa vs (Grifinória + Corvinal + Sonserina)
> 3. Corvinal vs (Grifinória + Lufa-Lufa + Sonserina)
> 4. Sonserina vs (Grifinória + Lufa-Lufa + Corvinal)
>
> Cada modelo gera uma probabilidade posterior $P(Y = c \mid X) = \sigma(\theta_c^T X)$. Na hora da inferência, aplicamos a regra de decisão **argmax**: o aluno é atribuído à casa cujo modelo estimou a maior probabilidade:
> $$\hat{y} = \arg\max_{c} \sigma(\theta_c^T X)$$

---

### P2: Por que a Sigmoide é necessária e por que não usamos apenas Regressão Linear?
> **R:** A saída de uma combinação linear $z = \theta^T X$ pode assumir qualquer valor real $(-\infty, +\infty)$, o que não representa uma probabilidade. A função Sigmoide mapeia qualquer número real para o intervalo aberto $(0, 1)$:
> $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
> Além disso, se usássemos MSE (Mean Squared Error) com a Sigmoide, a função de custo se tornaria **não-convexa**, cheia de mínimos locais. Usando a **Binary Cross-Entropy (Log-Loss)**, garantimos que a superfície de perda é estritamente **convexa**, assegurando convergência para o mínimo global.

---

### P3: Qual é a derivada da Log-Loss em relação aos pesos $\theta$?
> **R:** A perda para $m$ observações é:
> $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1 - y^{(i)}) \ln(1 - h_\theta(x^{(i)})) \right]$$
> Aplicando a regra da cadeia, sabendo que $\frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z))$, chegamos à elegante derivada analítica:
> $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
> Em notação matricial vetorizada:
> $$\nabla_\theta J = \frac{1}{m} X^T (h_\theta(X) - y)$$

---

### P4: Como você evitou Data Leakage (vazamento de dados) na normalização?
> **R:** No módulo `StandardScaler` (`src/preprocessing/scaler.py`), os parâmetros de média $\mu$ e desvio padrão $\sigma$ são calculados **estritamente sobre o dataset de treino** e salvos no arquivo `weights.json`.
> Durante o `logreg_predict.py`, o conjunto de teste **não recalcula** médias ou desvios; ele aplica exatamente os $(\mu_{\text{train}}, \sigma_{\text{train}})$ carregados de `weights.json`. Se houver valor ausente (NaN) no teste, imputamos a média de treino, o que resulta em $z = 0.0$ (valor neutro padronizado).

---

### P5: Qual a diferença prática entre Batch GD, SGD e Mini-Batch GD?
> **R:**
> - **Batch GD**: Calcula o gradiente somando o erro de todas as $m$ amostras antes de atualizar $\theta$. É determinístico e tem convergência suave, mas pode ser lento em datasets gigantes.
> - **SGD (Stochastic)**: Atualiza $\theta$ a cada amostra individual ($m=1$). É muito rápido por iteração e bom para escapar de platôs, mas oscila bastante em torno do mínimo.
> - **Mini-Batch GD**: Atualiza $\theta$ em blocos de tamanho $B$ (ex: 32 ou 64). É o melhor equilíbrio da indústria, pois aproveita a aceleração vetorial por blocos e estabiliza a variância dos gradientes.
