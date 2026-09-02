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
* **Analysis**: Finding collinear features (Pearson correlation $\approx 1.0$ or $-1.0$) prevents redundancy in our linear model weights.
* **Finding**: *(To be documented during visualization phase analysis)*.

## 3. Pair Plot (`pair_plot.py`)
* **Question**: *From this visualization, which features are you going to use for your logistic regression?*
* **Analysis**: Features with clear class boundaries and high variance between houses are prioritized; noisy or redundant features are dropped.
* **Finding**: *(To be documented during visualization phase analysis)*.
