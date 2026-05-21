# ADS 2026 Assignment 2 — Olist Brazilian E-Commerce Modeling

## Dataset
This project uses the Olist Brazilian E-Commerce Public Dataset.(Which was used in Assignment1)


## Main modeling tasks

### Part 1 — Regression
The regression task predicts actual delivery time in days. The main target is `delivery_time_days`.

Models evaluated:
- Linear Regression
- Kernel Regression using RBF Nystroem features + Ridge
- Ridge Regression
- LASSO Regression
- Decision Tree Regression as an optional benchmark

Primary metric: MAE, because the prediction error is directly interpretable as days of delivery error and is less dominated by extreme outliers than MSE.

### Part 2 — Binary Classification
The binary task predicts whether an order is delivered late relative to the estimated delivery date.

Target:
- 1 = late delivery
- 0 = on-time or early delivery

Models evaluated:
- Logistic Regression
- Linear SVM
- Kernel SVM
- KNN with K tuning
- Decision Tree with depth/leaf tuning
- Random Forest

Primary metric: F1, supported by ROC AUC and PR AUC, because late delivery can be imbalanced and both missed late orders and false warnings matter.

### Part 3 — Multiclass Classification
The multiclass task predicts four delivery-speed classes:
- very fast delivery
- fast delivery
- slow delivery
- very slow delivery

Models evaluated:
- Multiclass SVM
- Logistic Regression with One-vs-Rest
- Multinomial Logistic Regression
- KNN with K tuning
- Decision Tree with tuning
- AdaBoost
- XGBoost / LightGBM / CatBoost when installed

Primary metric: Macro F1, because all delivery-speed classes should matter even when their supports differ.

### Part 4 — Challenging Questions
The notebook answers the conceptual questions about bias--variance trade-off, kernel regression, L1/L2 regularization, MAPE, outliers, class imbalance, decision boundaries, KNN, decision-tree overfitting, tree feature selection, F1 variants, multilabel versus multiclass learning, precision--recall trade-off, ROC versus PR curves, and future improvements.

## Bonus additions
The notebook also includes:
- unified model comparison tables;
- model comparison visualizations;
- regression, binary, and multiclass error analysis;
- feature-importance and interpretability summaries;
- cross-validation stability summaries;
- training time and complexity comparison;
- an educational 2D decision-boundary demonstration.

## Reproducibility notes
The notebook uses a fixed `RANDOM_STATE` where applicable. Some models are intentionally capped or tuned with modest grids to remain executable on a normal laptop.
