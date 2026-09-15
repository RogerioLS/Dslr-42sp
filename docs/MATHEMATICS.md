# Mathematical Foundations of DSLR

## 1. Descriptive Statistics
To adhere strictly to 42's no-cheating policy, all statistics are implemented without high-level library helpers:

- **Count ($N$)**: Total valid non-NaN observations.
- **Mean ($\mu$)**:
  $$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$
- **Sample Standard Deviation ($\sigma$)**:
  $$\sigma = \sqrt{\frac{1}{N - 1}\sum_{i=1}^N (x_i - \mu)^2}$$
- **Percentiles ($25\%, 50\%, 75\%$)**:
  Given a sorted array $X_{sorted}$ of length $N$:
  $$\text{index} = (N - 1) \cdot p$$
  Let $i = \lfloor \text{index} \rfloor$ and $f = \text{index} - i$:
  $$\text{Percentile}(p) = X[i] + f \cdot (X[i+1] - X[i])$$
- **Sample Covariance**:
  $$\text{cov}(X, Y) = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu_X)(y_i - \mu_Y)$$
- **Pearson Correlation Coefficient**:
  $$r_{XY} = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^N (x_i - \mu_X)(y_i - \mu_Y)}{\sqrt{\sum_{i=1}^N (x_i - \mu_X)^2} \sqrt{\sum_{i=1}^N (y_i - \mu_Y)^2}}$$

---

## 2. Logistic Regression (One-vs-Rest)

### Sigmoid Hypothesis
$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### Binary Cross-Entropy Cost Function
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1 - y^{(i)}) \ln(1 - h_\theta(x^{(i)})) \right]$$

### Gradient Computation
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

### Feature Standardization (Z-Score)
To ensure stable and fast gradient descent convergence across features with different scales:
$$z = \frac{x - \mu}{\sigma}$$

During test inference, we strictly use the sample parameters $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ fitted on the training set to prevent data leakage. Missing values are imputed with $\mu_{\text{train}}$, yielding $z = 0.0$.

### Multiclass Decision Rule (One-vs-Rest Argmax)
For $K = 4$ classes (Gryffindor, Hufflepuff, Ravenclaw, Slytherin), we train $K$ independent binary classifiers parametrized by weight vectors $\theta_c$. For any test student sample $x$, the predicted class is chosen by maximum posterior probability:
$$\hat{y} = \arg\max_{c \in \mathcal{C}} P(Y = c \mid x) = \arg\max_{c \in \mathcal{C}} \sigma(\theta_c^T x)$$
where:
$$P(Y = c \mid x) = \frac{1}{1 + e^{-\theta_c^T x}}$$

---

## 3. Bonus Features & Optimization Algorithms

### 3.1 Extended Descriptive Statistics
- **Sample Variance ($s^2$)**:
  $$s^2 = \frac{1}{N - 1}\sum_{i=1}^N (x_i - \mu)^2$$
- **Interquartile Range ($IQR$)**:
  $$\text{IQR} = Q_3 - Q_1 = \text{Percentile}(0.75) - \text{Percentile}(0.25)$$
- **Sample Skewness ($g_1$)** (Fisher-Pearson unbiased 3rd standardized moment):
  $$g_1 = \frac{N}{(N - 1)(N - 2)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^3$$
- **Sample Excess Kurtosis ($g_2$)** (Fisher definition, normal distribution = 0):
  $$g_2 = \frac{N(N + 1)}{(N - 1)(N - 2)(N - 3)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^4 - \frac{3(N - 1)^2}{(N - 2)(N - 3)}$$

### 3.2 Optimization Methods Comparison

| Optimizer | Batch Size ($B$) | Parameter Update Frequency | Trade-off / Characteristic |
|---|---|---|---|
| **Batch GD** (`--method batch`) | $m$ (all samples) | Once per epoch | Deterministic, exact gradient, smooth monotonic loss decay. |
| **SGD** (`--method sgd`) | $1$ (single sample) | $m$ times per epoch | High stochasticity, escapes local minima, noisy loss oscillations. |
| **Mini-Batch GD** (`--method minibatch`) | $32$ / $64$ | $\lceil m / B \rceil$ times per epoch | Ideal sweet spot: SIMD vectorized matrix stability + stochastic escape. |

$$\theta := \theta - \alpha \cdot \frac{1}{|B|} \sum_{i \in B} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$
