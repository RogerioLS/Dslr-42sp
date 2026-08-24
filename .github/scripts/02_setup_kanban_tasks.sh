#!/bin/bash
# ==============================================================================
#           42 DSLR — STEP 2: SETUP KANBAN TASKS (IDEMPOTENT)
# ==============================================================================

set -e

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)

if [ -z "$REPO" ]; then
    echo "❌ Erro: Não foi possível detectar o repositório GitHub via gh cli."
    exit 1
fi

echo "🚀 [ETAPA 2] Populando o Kanban com as tasks no repositório: $REPO..."

# Função auxiliar para criar issue somente se ela não existir
create_issue_if_missing() {
    local title="$1"
    local milestone="$2"
    local labels="$3"
    local body="$4"

    # Checa se já existe uma issue aberta com esse título
    local existing=$(gh issue list --search "$title in:title" --json number -q '.[0].number' 2>/dev/null || true)

    if [ -n "$existing" ]; then
        echo "⏭️ Issue já existente (#$existing): $title"
    else
        echo "➕ Criando issue: $title"
        gh issue create \
            --title "$title" \
            --milestone "$milestone" \
            --label "$labels" \
            --body "$body"
    fi
}

# --- MILESTONE 1 ---

create_issue_if_missing \
  "[DSLR-01] Data Pipeline: Carregamento do CSV e Tratamento de NaNs" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: implementation,type: pedagogical,priority: high" \
  "## 🎯 Objetivo Didático
Aprender como estruturar a ingestão de dados tabulares e o manuseio de valores ausentes (*missing data/NaNs*) sem depender de métodos prontos de imputação.

## 📚 Conceito para Estudo em Dupla
Em Ciência de Dados, dados brutos quase nunca estão limpos. Alunos de Hogwarts podem ter faltado em exames (valores vazios no CSV). Precisamos identificar colunas puramente numéricas e ignorar ou tratar valores ausentes durante cálculos analíticos sem quebrar o algoritmo.

## 📝 Tarefas Técnicas
- [ ] Criar módulo \`src/analytics/loader.py\` ou parser no pandas/numpy para carregar o CSV.
- [ ] Separar colunas numéricas de colunas categóricas/metadados (\`Index\`, \`Hogwarts House\`, \`First Name\`, \`Last Name\`, \`Birthday\`, \`Best Hand\`).
- [ ] Implementar filtro para desconsiderar NaNs no cálculo de cada coluna individual.

## 🧪 Critérios de Aceite
- Extração de matriz ou arrays de features numéricas funcionais.
- Preservação da integridade posicional dos dados."

create_issue_if_missing \
  "[DSLR-02] Math from Scratch: Motor Estatístico (Count, Mean, Std, Min, Max)" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: math-heavy,type: implementation,priority: high" \
  "## 🎯 Objetivo Didático
Implementar as operações estatísticas fundamentais do zero em Python puro, respeitando a regra estrita de **No-Cheating** da 42.

## 📚 Conceito para Estudo em Dupla
- **Count ($N$)**: Total de valores não nulos.
- **Mean ($\mu$)**: $\mu = \frac{1}{N} \sum x_i$. Representa o ponto de equilíbrio dos dados.
- **Sample Variance ($\sigma^2$)**: $\sigma^2 = \frac{1}{N-1} \sum (x_i - \mu)^2$. Usa-se $N-1$ (correção de Bessel) porque estamos trabalhando com uma amostra.
- **Standard Deviation ($\sigma$)**: $\sigma = \sqrt{\sigma^2}$. Mede o grau de dispersão em torno da média.
- **Min / Max**: Menor e maior valor finito do conjunto.

## ⚠️ Regra 42 Anti-Cheating
É expressamente proibido usar \`df.describe()\`, \`df.mean()\`, \`df.std()\`, \`np.mean()\`, \`np.std()\`, etc.

## 📝 Tarefas Técnicas
- [ ] Implementar funções em \`src/analytics/statistics.py\`.
- [ ] Garantir complexidade de tempo linear $O(N)$ para média e variância.
- [ ] Validar que \`make norm\` aprova a implementação sem erros.

## 🧪 Critérios de Aceite
- Resultados idênticos aos do Pandas/Numpy com tolerância de $10^{-6}$."

create_issue_if_missing \
  "[DSLR-03] Quantile Interpolation: Cálculo de Percentis (25%, 50%, 75%)" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: math-heavy,type: implementation,priority: high" \
  "## 🎯 Objetivo Didático
Entender a matemática da ordenação e da interpolação linear para determinar quartis e mediana.

## 📚 Conceito para Estudo em Dupla
- **25% (Q1 - Primeiro Quartil)**: Valor abaixo do qual estão 25% dos dados.
- **50% (Q2 - Mediana)**: Ponto central que divide o conjunto em duas metades iguais.
- **75% (Q3 - Terceiro Quartil)**: Valor abaixo do qual estão 75% dos dados.
- **Fórmula de Interpolação Linear (Método 7 - padrão Pandas/Numpy)**:
  $$\text{index} = (N - 1) \cdot p$$
  $$\text{Percentil}(p) = X[\lfloor \text{index} \rfloor] + (\text{index} - \lfloor \text{index} \rfloor) \cdot (X[\lceil \text{index} \rceil] - X[\lfloor \text{index} \rfloor])$$

## 📝 Tarefas Técnicas
- [ ] Implementar algoritmo de ordenação ou usar \`sorted()\` em lista sem NaNs.
- [ ] Implementar a fórmula de interpolação linear.
- [ ] Cobrir casos de borda ($N=1$, valores idênticos, lista vazia).

## 🧪 Critérios de Aceite
- Testes unitários em \`tests/test_statistics.py\` validando valores de $25\%$, $50\%$ e $75\%$ contra o output oficial."

create_issue_if_missing \
  "[DSLR-04] CLI describe.py: Formatação Visual e Alinhamento de Tabela" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: implementation,priority: high" \
  "## 🎯 Objetivo Didático
Construir a interface de linha de comando oficial da primeira parte do subject e formatar o output no terminal de maneira profissional.

## 📚 Conceito para Estudo em Dupla
O subject exige que \`describe.py\` receba um arquivo CSV como parâmetro e exiba a tabela de todas as features numéricas com 6 casas decimais, alinhadas por colunas.

## 📝 Tarefas Técnicas
- [ ] Criar \`describe.py\` na raiz do repositório.
- [ ] Tratar argumentos da CLI (\`sys.argv\`) com mensagens de erro claras se o arquivo não existir.
- [ ] Formatar o cabeçalho e as linhas com \`{:>15.6f}\` para alinhamento uniforme.
- [ ] Testar com \`make describe\`.

## 🧪 Critérios de Aceite
- Execução: \`python3 describe.py datasets/dataset_train.csv\`.
- Saída visual idêntica à página 6 do PDF do subject."

# --- MILESTONE 2 ---

create_issue_if_missing \
  "[DSLR-05] Histogram Analysis: Identificação do Curso com Distribuição Homogênea" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,type: pedagogical,priority: medium" \
  "## 🎯 Objetivo Didático
Compreender como histogramas revelam a distribuição estatística de notas de cada casa de Hogwarts.

## 📚 Pergunta do Subject
*Which Hogwarts course has a homogeneous score distribution between all four houses?*

## 📚 Conceito para Estudo em Dupla
Uma matéria onde Gryffindor, Hufflepuff, Ravenclaw e Slytherin possuem exatamente a mesma distribuição (mesma média e dispersão) não ajuda o Chapéu Seletor a separar quem é de qual casa. Essa feature tem poder discriminativo quase nulo.

## 📝 Tarefas Técnicas
- [ ] Criar \`histogram.py\` na raiz.
- [ ] Gerar gráficos sobrepostos das 4 casas para cada uma das matérias de Hogwarts (com cores temáticas: Vermelho, Amarelo, Azul, Verde).
- [ ] Documentar no \`docs/DATA_VISUALIZATION.md\` qual matéria é homogênea e por quê.

## 🧪 Critérios de Aceite
- Gráfico exibido na tela de forma legível com legenda das 4 casas.
- Resposta fundamentada documentada."

create_issue_if_missing \
  "[DSLR-06] Bivariate Analysis: Identificação de Features Correlacionadas (Scatter Plot)" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,type: pedagogical,priority: medium" \
  "## 🎯 Objetivo Didático
Analisar a correlação bivariada entre pares de matérias para identificar redundâncias no dataset.

## 📚 Pergunta do Subject
*What are the two features that are similar?*

## 📚 Conceito para Estudo em Dupla
Se duas matérias possuem uma relação linear quase perfeita (correlação de Pearson $\approx 1.0$), incluir ambas no modelo de Regressão Logística adiciona multicolinearidade sem trazer informação nova.

## 📝 Tarefas Técnicas
- [ ] Criar \`scatter_plot.py\` na raiz.
- [ ] Calcular matriz de correlação de Pearson para guiar a busca visual.
- [ ] Plotar o gráfico de dispersão com os pontos das 4 casas coloridos.
- [ ] Documentar o par de features correlacionadas em \`docs/DATA_VISUALIZATION.md\`.

## 🧪 Critérios de Aceite
- Gráfico de dispersão claro mostrando a correlação evidente entre as duas matérias."

create_issue_if_missing \
  "[DSLR-07] Multivariate Matrix: Pair Plot e Seleção de Features" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,type: pedagogical,priority: medium" \
  "## 🎯 Objetivo Didático
Gerar uma matriz de dispersão completa (*pair plot / scatter matrix*) para fundamentar matematicamente quais features serão usadas no treinamento.

## 📚 Pergunta do Subject
*From this visualization, which features are you going to use for your logistic regression?*

## 📚 Conceito para Estudo em Dupla
Um Pair Plot plota todas as combinações de 2 a 2 matérias (gráficos de dispersão fora da diagonal) e a densidade/histograma na diagonal principal. As melhores features são aquelas onde as 4 nuvens de cores estão claramente separadas no espaço.

## 📝 Tarefas Técnicas
- [ ] Criar \`pair_plot.py\` na raiz utilizando Seaborn/Matplotlib.
- [ ] Documentar a lista final de features selecionadas justificando a exclusão das matérias homogêneas e redundantes.

## 🧪 Critérios de Aceite
- Matriz completa gerada sem cortes de escala.
- Relatório de justificativa em \`docs/DATA_VISUALIZATION.md\`."

# --- MILESTONE 3 ---

create_issue_if_missing \
  "[DSLR-08] Preprocessing: Normalização Z-Score (StandardScaler)" \
  "03. Logistic Regression Engine & Math" \
  "area: preprocessing,area: model,type: math-heavy,type: implementation,priority: high" \
  "## 🎯 Objetivo Didático
Entender a importância do escalonamento de features para o algoritmo de Gradiente Descendente.

## 📚 Conceito para Estudo em Dupla
Se uma matéria varia de $0$ a $10$ e outra varia de $-1000$ a $+1000$, as curvas de nível da função de custo $J(\theta)$ tornam-se elipses extremamente estreitas. O gradiente oscilará em zigue-zague e demorará muito para convergir.
Com **StandardScaler (Z-score)**:
$$z = \frac{x - \mu}{\sigma}$$
Todas as features passam a ter média $0$ e desvio padrão $1$, tornando o espaço de perda esférico e a descida do gradiente estável.

## 📝 Tarefas Técnicas
- [ ] Criar classe \`StandardScaler\` artesanal em \`src/preprocessing/scaler.py\`.
- [ ] Implementar métodos \`fit(X)\`, \`transform(X)\` e \`fit_transform(X)\`.
- [ ] Salvar as médias $\mu$ e desvios $\sigma$ do treino para aplicar identicamente no teste (sem data leakage!).

## 🧪 Critérios de Aceite
- Média pós-transformação $\approx 0$ e desvio padrão $\approx 1$ em todas as colunas numéricas."

create_issue_if_missing \
  "[DSLR-09] Core Math: Função Sigmoide, Log-Loss e Gradiente Analítico" \
  "03. Logistic Regression Engine & Math" \
  "area: model,type: math-heavy,type: implementation,priority: high" \
  "## 🎯 Objetivo Didático
Implementar a formulação matemática da Regressão Logística em formato matricial/vetorizado.

## 📚 Fórmulas Fundamentais
1. **Hipótese Sigmoide**:
   $$g(z) = \frac{1}{1 + e^{-z}}, \quad h_\theta(X) = g(X\theta)$$
2. **Custo (Binary Cross-Entropy / Log-Loss)**:
   $$J(\theta) = -\frac{1}{m} \left[ y^T \log(h_\theta(X) + \epsilon) + (1 - y)^T \log(1 - h_\theta(X) + \epsilon) \right]$$
3. **Gradiente do Custo**:
   $$\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)$$

## 📝 Tarefas Técnicas
- [ ] Implementar funções vetorizadas com Numpy em \`src/model/logistic_regression.py\`.
- [ ] Adicionar estabilidade numérica ($\epsilon = 10^{-15}$) para evitar $\log(0) = -\infty$.
- [ ] Adicionar termo de bias (coluna de $1$s na matriz $X$).

## 🧪 Critérios de Aceite
- Testes unitários validando valores da sigmoide e dimensionalidade das matrizes."

create_issue_if_missing \
  "[DSLR-10] Training Engine: One-vs-Rest (OvR) e Batch Gradient Descent" \
  "03. Logistic Regression Engine & Math" \
  "area: model,type: implementation,type: math-heavy,priority: high" \
  "## 🎯 Objetivo Didático
Construir o pipeline de treinamento supervisionado multiclasse para as 4 casas de Hogwarts.

## 📚 Conceito para Estudo em Dupla
Como a Regressão Logística é naturalmente binária ($0$ ou $1$), o método **One-vs-Rest (OvR)** treina 4 modelos separados:
1. Gryffindor ($1$) vs Outras ($0$)
2. Hufflepuff ($1$) vs Outras ($0$)
3. Ravenclaw ($1$) vs Outras ($0$)
4. Slytherin ($1$) vs Outras ($0$)

## 📝 Tarefas Técnicas
- [ ] Criar executável \`logreg_train.py\` na raiz.
- [ ] Implementar loop de Batch Gradient Descent com hiperparâmetros ajustáveis (learning rate $\alpha$, epochs).
- [ ] Salvar os pesos $\theta$ das 4 casas e parâmetros do scaler em \`weights.json\`.
- [ ] Exibir a evolução da perda ($J(\theta)$) a cada época para checar convergência.

## 🧪 Critérios de Aceite
- Execução: \`python3 logreg_train.py datasets/dataset_train.csv\`.
- Geração bem-sucedida do arquivo de pesos."

create_issue_if_missing \
  "[DSLR-11] Inference Engine: Predição Multiclasse e houses.csv" \
  "03. Logistic Regression Engine & Math" \
  "area: model,type: implementation,type: defense,priority: high" \
  "## 🎯 Objetivo Didático
Carregar os pesos treinados, classificar novos alunos do \`dataset_test.csv\` e validar a meta de acurácia $\ge 98\%$.

## 📚 Regra de Decisão
Para cada aluno $x$, calculamos a probabilidade de pertencer a cada casa:
$$\hat{y} = \arg\max_{c \in \{\text{Gryffindor}, \text{Hufflepuff}, \text{Ravenclaw}, \text{Slytherin}\}} h_{\theta_c}(x)$$

## 📝 Tarefas Técnicas
- [ ] Criar \`logreg_predict.py\` na raiz recebendo \`dataset_test.csv\` e \`weights.json\`.
- [ ] Normalizar o conjunto de teste usando $\mu$ e $\sigma$ aprendidos no treino.
- [ ] Gerar \`houses.csv\` no formato estrito do subject:
  \`\`\`csv
  Index,Hogwarts House
  0,Gryffindor
  1,Hufflepuff
  ...
  \`\`\`
- [ ] Validar acurácia com \`make evaluate\` ou Scikit-Learn accuracy_score.

## 🧪 Critérios de Aceite
- Acurácia comprovada $\ge 98.0\%$ no conjunto de teste."

# --- MILESTONE 4 ---

create_issue_if_missing \
  "[DSLR-12] Bonus: Otimizadores (SGD / Mini-Batch GD) & Estatísticas Extras" \
  "04. Defense Readiness & Bonuses" \
  "area: model,type: bonus,type: implementation,priority: low" \
  "## 🎯 Objetivo Didático
Explorar variações do algoritmo de otimização e enriquecer a análise estatística.

## 📚 Conceitos dos Bônus
- **SGD (Stochastic Gradient Descent)**: Atualiza $\theta$ a cada amostra individual ($m=1$). Muito rápido, mas com gradiente ruidoso.
- **Mini-Batch Gradient Descent**: Atualiza $\theta$ em lotes pequenos (ex: batch de 32 ou 64). Equilibra velocidade e estabilidade vetorial.
- **Métricas Extras no describe.py**: Skewness (assimetria), Kurtosis (curtose), IQR, Variância e contagem de NaNs.

## 📝 Tarefas Técnicas
- [ ] Adicionar flag \`--method batch/sgd/minibatch\` no \`logreg_train.py\`.
- [ ] Implementar estatísticas adicionais no \`src/analytics/statistics.py\`.

## 🧪 Critérios de Aceite
- Comparação de tempo de execução e curvas de convergência documentadas."

create_issue_if_missing \
  "[DSLR-13] Peer Defense Simulator: Mock Interview & Perguntas da 42" \
  "04. Defense Readiness & Bonuses" \
  "area: defense,type: pedagogical,type: defense,priority: high" \
  "## 🎯 Objetivo Didático
Preparar a dupla para a avaliação presencial da 42 através de uma rodada completa de perguntas e respostas.

## 📚 Tópicos Obrigatórios na Defesa
1. **No-Cheating**: Mostrar o código do \`describe.py\` provando que nenhuma função do Pandas/Numpy foi usada nos cálculos.
2. **Visualização**: Explicar por que a matéria homogênea foi descartada e apontar o par de matérias correlacionadas.
3. **Matemática da Regressão**: Escrever ou explicar a derivada da Log-Loss e o papel da Sigmoide.
4. **Demonstração Prática**: Executar \`make audit\`, treinar o modelo e gerar as predições ao vivo na frente do avaliador.

## 📝 Tarefas Técnicas
- [ ] Fazer uma sessão de treino em dupla usando o roteiro em \`docs/PEER_EVALUATION_GUIDE.md\`.
- [ ] Revisar todas as docstrings e tipagens.

## 🧪 Critérios de Aceite
- Ambos os membros da dupla aptos a explicar qualquer linha de código do repositório."

echo "🎉 Todas as Tasks foram verificadas e criadas no GitHub!"
