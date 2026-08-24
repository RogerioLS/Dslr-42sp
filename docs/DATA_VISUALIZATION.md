# Data Visualization Guide & Analysis

This document records the visual findings corresponding to the three mandatory exploration questions:

## 1. Histogram (`histogram.py`)
* **Question**: *Which Hogwarts course has a homogeneous score distribution between all four houses?*
* **Analysis**: Features with identical bell curves/distributions across Gryffindor, Hufflepuff, Ravenclaw, and Slytherin provide no discriminative power for classification.
* **Finding**: *(To be documented during visualization phase analysis)*.

## 2. Scatter Plot (`scatter_plot.py`)
* **Question**: *What are the two features that are similar?*
* **Analysis**: Finding collinear features (Pearson correlation $\approx 1.0$ or $-1.0$) prevents redundancy in our linear model weights.
* **Finding**: *(To be documented during visualization phase analysis)*.

## 3. Pair Plot (`pair_plot.py`)
* **Question**: *From this visualization, which features are you going to use for your logistic regression?*
* **Analysis**: Features with clear class boundaries and high variance between houses are prioritized; noisy or redundant features are dropped.
* **Finding**: *(To be documented during visualization phase analysis)*.
