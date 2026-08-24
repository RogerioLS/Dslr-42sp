# Mathematics & Algorithm Specifications

## 1. Descriptive Statistics Formulas
- **Mean**: $\mu = \frac{1}{N} \sum_{i=1}^N x_i$
- **Variance & Std**: $\sigma^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)^2$, $\sigma = \sqrt{\sigma^2}$ (sample standard deviation)
- **Percentiles**: Linear interpolation between sorted rank indices $P = (N - 1) \cdot p$.

## 2. Logistic Regression
- **Hypothesis (Sigmoid)**:
  $$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
- **Cost Function (Binary Cross-Entropy / Log-Loss)**:
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$
- **Gradient**:
  $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$
- **Batch Gradient Descent Update**:
  $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

## 3. Multi-class (One-vs-Rest)
Train $K=4$ binary classifiers (one for each house: Gryffindor, Hufflepuff, Ravenclaw, Slytherin).
Predict class $k = \arg\max_{c \in \{1..4\}} h_{\theta_c}(x)$.
