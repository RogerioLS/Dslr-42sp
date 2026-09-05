# Data Visualization Guide & Analysis

This document records the visual findings corresponding to the three mandatory exploration questions:

## 1. Histogram (`histogram.py`)
* **Question**: *Which Hogwarts course has a homogeneous score distribution between all four houses?*
* **Analysis**: Features with identical bell curves/distributions across Gryffindor, Hufflepuff, Ravenclaw, and Slytherin provide no discriminative power for classification.
* **Finding**: **Arithmancy** has a homogeneous score distribution across all four houses.
  Visually, the 13-course grid shows most subjects splitting into two or more clearly
  separated humps by house (e.g. Astronomy, Herbology, Transfiguration, Charms, Flying),
  while Arithmancy and Care of Magical Creatures both show heavily overlapping,
  single-hump distributions. Comparing the two numerically (mean difference between
  houses relative to each house's standard deviation, since the two courses use very
  different score scales) breaks the tie:

  | Course | Mean range across houses | Avg. std | Relative spread |
  |---|---|---|---|
  | Arithmancy | 49,122 – 50,249 | ≈ 16,578 | **≈ 6.8%** |
  | Care of Magical Creatures | -0.14 – 0.00 | ≈ 0.97 | ≈ 14.4% |

  Arithmancy's house means are proportionally closer together, confirming it as the
  more homogeneous course — even though Care of Magical Creatures looks like a tighter
  single blob at first glance, since its raw score scale (-1 to 1) makes small absolute
  differences look visually smaller than they proportionally are.

## 2. Scatter Plot (`scatter_plot.py`)
* **Question**: *What are the two features that are similar?*
* **Analysis**: Finding collinear features (Pearson correlation $|r| \approx 1.0$) prevents redundancy and multicollinearity in our linear model weights.
* **Finding**: **Astronomy** and **Defense Against the Dark Arts** are the two similar features.
  Across all $\binom{13}{2} = 78$ pairs of Hogwarts courses, they exhibit a **perfect negative linear correlation** ($r = -1.000000$, $|r| = 1.000000$).
  Inspection of raw values reveals an exact functional relationship:
  $$\text{Defense Against the Dark Arts} = -0.01 \times \text{Astronomy}$$

  Top 5 absolute correlations across all courses:

  | Feature 1 | Feature 2 | Pearson $r$ | $\|r\|$ |
  |---|---|---|---|
  | **Astronomy** | **Defense Against the Dark Arts** | **-1.000000** | **1.000000** |
  | History of Magic | Flying | -0.896283 | 0.896283 |
  | Transfiguration | Flying | -0.873673 | 0.873673 |
  | History of Magic | Transfiguration | +0.849203 | 0.849203 |
  | Muggle Studies | Charms | +0.847607 | 0.847607 |

  **ML Modeling Takeaway**: Including both *Astronomy* and *Defense Against the Dark Arts* in Logistic Regression would add strict multicollinearity without introducing new variance or predictive signal. One of the two must be dropped during feature selection.

## 3. Pair Plot (`pair_plot.py`)
* **Question**: *From this visualization, which features are you going to use for your logistic regression?*
* **Analysis**: A pair plot exposes both the single-feature class separability along the diagonal (histograms) and bivariate cluster separation off the diagonal (scatter matrix). Optimal features for Logistic Regression must exhibit distinct, separable clusters with minimal overlap between houses, while avoiding multicollinearity and uninformative noise.
* **Finding**: We select a subset of **9 high-discrimination features** and exclude **4 uninformative or redundant features**.

  ### Feature Separability Ranking (ANOVA F-statistic)
  Quantifying between-house variance relative to within-house variance ($F = \frac{MS_{between}}{MS_{within}}$) provides exact numerical support for the visual clusters observed in the pair plot:

  | Course | F-statistic | Pair Plot Cluster Status | Decision | Rationale |
  |---|---|---|---|---|
  | **Defense Against the Dark Arts** | 4,145.73 | Distinct house separation | **EXCLUDE** | Collinear with Astronomy ($r = -1.0000$). Redundant. |
  | **Astronomy** | 4,127.73 | Distinct house separation | **KEEP** | High discriminative power between all 4 houses. |
  | **Charms** | 3,663.52 | Strong cluster boundaries | **KEEP** | Separates Ravenclaw/Slytherin distinctly. |
  | **Ancient Runes** | 3,079.46 | Strong cluster boundaries | **KEEP** | Sharp multimodal house distinction. |
  | **Divination** | 2,792.18 | Clear separation | **KEEP** | Clear separation between Gryffindor and Slytherin. |
  | **Herbology** | 2,577.28 | Clear separation | **KEEP** | Isolates Hufflepuff strongly. |
  | **Transfiguration** | 2,425.80 | Clear separation | **KEEP** | Excellent bivariate contrast with Charms/Flying. |
  | **Flying** | 2,356.53 | Clear separation | **KEEP** | Distinguishes Gryffindor and Ravenclaw. |
  | **Muggle Studies** | 2,277.63 | Clear separation | **KEEP** | Clear separation profile across houses. |
  | **History of Magic** | 2,025.29 | Clear separation | **KEEP** | Strong contrast in pairwise scatter plots. |
  | **Potions** | 491.07 | Heavily overlapping | **EXCLUDE** | Low discriminative separation relative to top tier. |
  | **Care of Magical Creatures** | 1.57 | Single overlapping bell curve | **EXCLUDE** | Negligible between-house variance; noise. |
  | **Arithmancy** | 0.38 | Single overlapping bell curve | **EXCLUDE** | Perfectly homogeneous across houses; zero signal. |

  ### Final Selected Feature Subset
  $$\mathcal{F}_{selected} = \{\text{Astronomy}, \text{Herbology}, \text{Divination}, \text{Muggle Studies}, \text{Ancient Runes}, \text{History of Magic}, \text{Transfiguration}, \text{Charms}, \text{Flying}\}$$

  This 9-feature subset eliminates multicollinearity, strips noise and zero-variance signals, and provides well-conditioned linear decision boundaries capable of achieving the required $\ge 98\%$ classification accuracy.
