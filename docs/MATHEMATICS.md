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
