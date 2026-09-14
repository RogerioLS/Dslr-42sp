<table width="100%">
  <tr></tr>
  <tr>
    <td colspan="2" align="center">
      <h1>🧙‍♂️ 42 DSLR — DATA SCIENCE × LOGISTIC REGRESSION</h1>
      <h3>Hogwarts Sorting Hat Multi-Class Classifier from First Mathematical Principles</h3>
      <p align="center">
        <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white" alt="Python 3.10+"/>
        <img src="https://img.shields.io/badge/Norm_42-100%25_Compliant-2ea44f?style=flat&logo=checkmarx&logoColor=white" alt="Norm 42"/>
        <img src="https://img.shields.io/badge/Anti--Cheating-AST_Audited-8957e5?style=flat&logo=shield&logoColor=white" alt="Anti Cheating"/>
        <img src="https://img.shields.io/badge/Target_Precision-Accuracy_%E2%89%A5_98%25-success?style=flat" alt="Precision"/>
        <img src="https://img.shields.io/badge/Tests-103_Passed-success?style=flat&logo=pytest&logoColor=white" alt="Tests"/>
        <img src="https://img.shields.io/badge/License-MIT-purple?style=flat" alt="License"/>
      </p>
      <p align="center">
        <a href="https://github.com/RogerioLS/Dslr-42sp/actions" target="_blank">
          <img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-0969da?style=flat&logo=githubactions&logoColor=white" alt="CI/CD"/>
        </a>
        <img src="https://img.shields.io/badge/%C3%89cole_42-S%C3%A3o_Paulo_🇧🇷-000000?style=flat&logo=42&logoColor=white" alt="42 SP"/>
      </p>
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td width="55%">
      <p>
        🎯 <strong>Mission & Objective:</strong><br>
        Implementation of a full data science pipeline and multi-class <strong>Logistic Regression model from raw mathematical first principles</strong> to sort Hogwarts students into four houses (<em>Gryffindor</em>, <em>Hufflepuff</em>, <em>Ravenclaw</em>, <em>Slytherin</em>) with test accuracy $\ge 98.0\%$.
      </p>
      <p>
        🛡️ <strong>Zero-Cheating Architecture:</strong><br>
        Strictly zero usage of built-in statistical or ML solvers (<code>df.describe()</code>, <code>np.mean()</code>, <code>np.std()</code>, <code>np.percentile()</code>, <code>sklearn.*</code>). Every metric, gradient, cost surface, and parameter update step is derived analytically and coded by hand.
      </p>
      <p>
        ⚡ <strong>Numerical Stability & Feature Scaling:</strong><br>
        Features exhibit vastly different numerical scales (from $-1000$ to $+100,000$). Mitigates vanishing/exploding gradients via handcrafted <strong>StandardScaler ($Z$-Score)</strong> normalization $(\mu, \sigma)$ calculated strictly on training data and persisted in <code>weights.json</code> to prevent data leakage.
      </p>
      <p>
        🧙 <strong>One-vs-Rest (OvR) Classification:</strong><br>
        Coordinates 4 independent binary classifiers trained via Batch Gradient Descent, producing calibrated probability distributions resolved via argmax: $\hat{y} = \arg\max_{k} P(y=k \mid X)$.
      </p>
    </td>
    <td width="45%" align="center">
      <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDN1ZmkyejkzNXJ0anZyd21mNGdwYWpubzk1ZzcxOXN4ZW5uMm1iNCZlcD12MV9naWZzX3JlbGF0ZWQmY3Q9Zw/Tl2AK8HOHj7SU/giphy.gif" alt="Harry Potter Sorting Hat" width="100%" style="border-radius: 6px; max-width: 360px;"/>
      <p align="center">
        <sub><em>"There's nothing hidden in your head the Sorting Hat can't see..."<br>Multi-Class One-vs-Rest Logistic Regression</em></sub>
      </p>
    </td>
  </tr>
</table>

<table width="100%" align="center">
  <tr></tr>
  <tr>
    <td colspan="3" align="center">
      <h3>📐 Mathematical Formulation & First Principles</h3>
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td width="25%"><strong>Concept</strong></td>
    <td width="40%"><strong>Mathematical Formulation</strong></td>
    <td width="35%"><strong>Computational Implementation</strong></td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>Z-Score Standardization</strong></td>
    <td>$$z = \frac{x - \mu}{\sigma}, \quad \sigma = \sqrt{\frac{1}{m-1} \sum_{i=1}^m (x_i - \bar{x})^2}$$</td>
    <td>Handcrafted sample statistics computed with zero built-in functions in <code>StandardScaler</code>.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>Sigmoid Hypothesis</strong></td>
    <td>$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$</td>
    <td>Numerically stable vectorized implementation clipping $z \in [-500, 500]$ avoiding overflow.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>Cost Function (Log-Loss)</strong></td>
    <td>$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1 - y^{(i)}) \ln(1 - h_\theta(x^{(i)})) \right]$$</td>
    <td>Binary Cross-Entropy Loss with $\varepsilon = 10^{-15}$ clipping preventing $\ln(0)$ divergence.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>Analytical Gradients</strong></td>
    <td>$$\nabla_\theta J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)$$</td>
    <td>Exact matrix derivative computed in $O(m \cdot n)$ vector operations with zero finite differences.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>Batch Gradient Descent</strong></td>
    <td>$$\theta := \theta - \alpha \nabla_\theta J(\theta)$$</td>
    <td>Synchronous vector parameter updates over $N$ epochs with learning rate $\alpha = 0.5$.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><strong>One-vs-Rest Decision Rule</strong></td>
    <td>$$\hat{y} = \arg\max_{k \in \{0, 1, 2, 3\}} h_{\theta^{(k)}}(x)$$</td>
    <td>Evaluates 4 probability scores simultaneously and selects the dominant house assignment.</td>
  </tr>
</table>

<table width="100%">
  <tr></tr>
  <tr>
    <td colspan="4" align="center">
      <h3>📦 42 Deliverables Contract & Executables</h3>
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td width="20%"><strong>Executable</strong></td>
    <td width="15%"><strong>Category</strong></td>
    <td width="30%"><strong>Terminal Command</strong></td>
    <td width="35%"><strong>Scope & Expected Output</strong></td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>describe.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make describe</code></td>
    <td>Lê dataset CSV e renderiza tabela estatística completa (Count, Mean, Std, Min, 25%, 50%, 75%, Max) sem Pandas.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>histogram.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make histogram</code></td>
    <td>Gera histogramas sobrepostos das 4 casas para identificar o curso homogêneo (<em>Care of Magical Creatures</em>).</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>scatter_plot.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make scatter</code></td>
    <td>Plota diagramas de dispersão bivariados para evidenciar as duas matérias mais semelhantes/correlacionadas.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>pair_plot.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make pairplot</code></td>
    <td>Matriz completa de pair plots colorida por casa, servindo como base visual para seleção de features.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>logreg_train.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make train</code></td>
    <td>Treina os 4 modelos binários OvR via Batch Gradient Descent e exporta o arquivo <code>weights.json</code>.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>logreg_predict.py</code></td>
    <td><span style="color: #2ea44f;">Mandatory</span></td>
    <td><code>make predict</code></td>
    <td>Carrega <code>weights.json</code>, aplica scaler e gera <code>houses.csv</code> com as predições finais ($\ge 98\%$ acurácia).</td>
  </tr>
</table>

<table width="100%">
  <tr></tr>
  <tr>
    <td colspan="3" align="center">
      <h3>🕹️ Command Center & Quality Gates (Makefile)</h3>
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td width="25%"><strong>Target</strong></td>
    <td width="25%"><strong>Category</strong></td>
    <td width="50%"><strong>Action & Quality Check</strong></td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make help</code></td>
    <td>Developer Experience</td>
    <td>Exibe o menu interativo com todos os comandos disponíveis no Command Center.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make install</code></td>
    <td>Setup & Tooling</td>
    <td>Instala dependências do projeto em modo editável e configura os git hooks em <code>.githooks/</code>.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make audit</code></td>
    <td>Quality Gate 42</td>
    <td>Validação completa: compilação Python 3.10, auditor AST Anti-Cheating e 100% dos testes unitários.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make check</code></td>
    <td>Sanity Check</td>
    <td>Bateria rigorosa de linters: Norm Check, Black, isort, flake8, ruff e scanner de secrets.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make test</code></td>
    <td>Automated Tests</td>
    <td>Executa toda a suíte de testes unitários e de integração (103+ testes automatizados).</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make train</code></td>
    <td>Model Training</td>
    <td>Executa o pipeline completo de treinamento do modelo e salva <code>weights.json</code>.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make predict</code></td>
    <td>Inference</td>
    <td>Executa inferência sobre <code>datasets/dataset_test.csv</code> e gera <code>houses.csv</code>.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make evaluate</code></td>
    <td>Evaluation</td>
    <td>Calcula a acurácia de predição do modelo contra o limiar obrigatório de 98%.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make clean</code></td>
    <td>Maintenance</td>
    <td>Remove caches temporários (<code>__pycache__</code>, <code>.pytest_cache</code>, <code>houses.csv</code>).</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>make pr</code></td>
    <td>Git Automation</td>
    <td>Publica Pull Request automatizado no GitHub vinculando a issue correspondente.</td>
  </tr>
</table>

<table width="100%">
  <tr></tr>
  <tr>
    <td colspan="2" align="center">
      <h3>🏛️ Repository Architecture & Engineering Modules</h3>
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td width="35%"><strong>Module / Directory</strong></td>
    <td width="65%"><strong>Architectural Responsibility</strong></td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>src/analytics/</code></td>
    <td>Estatística matemática handcrafted: <code>loader.py</code> e <code>statistics.py</code> com zero funções de biblioteca.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>src/preprocessing/</code></td>
    <td>Tratamento de dados e normalizador <code>StandardScaler</code> com serialização JSON para prevenir vazamento de dados.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>src/model/</code></td>
    <td>Núcleo de Machine Learning: <code>logistic_regression.py</code> (gradientes, sigmoide, BCE loss) e <code>multiclass.py</code> (OvR).</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>src/visualization/</code></td>
    <td>Módulos gráficos: <code>histogram.py</code>, <code>scatter.py</code> e <code>pair.py</code> com paleta oficial das casas de Hogwarts.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>tests/</code></td>
    <td>Pirâmide de testes unitários e de integração validando axiomas matemáticos e contratos de linha de comando.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>scripts/</code></td>
    <td>Auditor AST Anti-Cheating (<code>norm_check.py</code>), avaliador de acurácia, geradores de relatório e automação de PR.</td>
  </tr>
  <tr></tr>
  <tr>
    <td><code>.agents/</code></td>
    <td>Regras de governança operacional e padrões estritos da École 42 para assistentes autônomos.</td>
  </tr>
</table>

<a href="#"><img align='right' src='https://raw.githubusercontent.com/RogerioLS/RogerioLS/main/foto_little.png' width='55'></a>
