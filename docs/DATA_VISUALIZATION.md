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
* **Analysis**: Features with clear class boundaries and high variance between houses are prioritized; noisy or redundant features are dropped.
* **Finding**: *(To be documented during visualization phase analysis)*.
