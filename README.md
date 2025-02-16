# Machine Learning Workshop – Practical notebooks for learning ML

## Overview

This repository contains a collection of Jupyter notebooks that cover various topics in machine learning, including classification, clustering, unsupervised learning, and statistical analysis. The notebooks feature hands-on implementations of popular algorithms, data preprocessing techniques, model evaluation, and hyperparameter tuning using real-world datasets from the UCI Machine Learning Repository.

Each notebook focuses on a specific machine learning task, providing insights into data exploration, model training, evaluation metrics, and visualization techniques.

**Objective**: Provide a comprehensive guide to understanding and implementing machine learning algorithms. \
**Audience**: Designed for students and professionals with a basic understanding of programming and statistics.

## Notebooks

### 1. Statistical Analysis of IRIS, WINE, and GLASS Datasets

**Description:**

- Performs statistical analysis on three well-known datasets: **IRIS**, **WINE**, and **GLASS**.
- Applies Principal Component Analysis (PCA) for dimensionality reduction.
- Visualizes results in 2D and 3D plots to uncover patterns in the datasets.

**Datasets:** [IRIS](https://archive.ics.uci.edu/dataset/53/iris), [WINE](https://archive.ics.uci.edu/dataset/109/wine) and [GLASS](https://archive.ics.uci.edu/dataset/109/wine)

### 2. Polish Companies Bankruptcy Prediction

**Description:**

- Builds a classification or clustering model using the **Polish Companies Bankruptcy** dataset.
- Covers data preprocessing steps such as standardization, outlier detection, and dimensionality reduction using PCA and LLE.
- Compares different preprocessing techniques for training a **Random Forest classifier**.

**Dataset:** [Polish Companies Bankruptcy](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

### 3. k-Nearest Neighbors (k-NN) Classification Analysis

**Description:**

- Investigates the performance of the k-NN classification algorithm on **IRIS**, **WINE**, and **GLASS** datasets.
- Evaluates classification quality using **Precision, Recall, F-score, Accuracy, and MCC**.
- Implements cross-validation with different n-fold sizes and a balanced stratified approach.

**Datasets:** [IRIS](https://archive.ics.uci.edu/dataset/53/iris), [WINE](https://archive.ics.uci.edu/dataset/109/wine) and [GLASS](https://archive.ics.uci.edu/dataset/109/wine)

### 4. Decision Tree Classification (CART Algorithm)

**Description:**

- Focuses on training and evaluating **decision tree classifiers**.
- Covers hyperparameter tuning including max depth, criterion, min samples leaf, and class weight.
- Provides visualizations to understand the impact of hyperparameters on model performance.

**Dataset:** [Secondary Mushroom](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

### 5. Clustering with K-Means and DBSCAN

**Description:**

- Introduces clustering using **K-Means** and **DBSCAN**.
- Uses the **Abalone** dataset to demonstrate clustering techniques.
- Evaluates performance using **Silhouette score** and **Davies-Bouldin index**.
- Analyzes the impact of key hyperparameters on clustering effectiveness.

**Dataset:** [Abalone](https://archive.ics.uci.edu/dataset/1/abalone)

### 6. Credit Approval Modeling

**Description:**

- Analyzes and models the **Credit Approval** dataset.
- Implements multiple machine learning models including **Random Forest, XGBoost, LightGBM, CatBoost, and Histogram-based Gradient Boosting**.
- Optimizes hyperparameters using Bayesian optimization with Optuna.
- Applies advanced techniques like monotonicity and interaction constraints to improve model performance.

**Dataset:** [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval)

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/machine-learning-notebooks.git
   cd machine-learning-notebooks
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Navigate to the notebook of interest and start exploring!

## License

This repository is licensed under the MIT License.
