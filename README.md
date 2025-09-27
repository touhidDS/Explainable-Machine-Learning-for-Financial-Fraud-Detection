# Credit Card Fraud Detection

## Overview
This repository contains a Jupyter Notebook for detecting credit card fraud using a dataset from Google Drive. The notebook handles class imbalance with SMOTE, trains models (Logistic Regression, Random Forest, XGBoost, Neural Network), evaluates performance, and uses SHAP for explainability, selecting the best model based on ROC AUC.

## Dataset
- **Source**: `/content/drive/MyDrive/creditcard.csv` (loaded via Google Colab)
- **Features**: 30 columns including `Time`, `V1` to `V28` (PCA-transformed), `Amount`.
- **Target**: `Class` (binary: 0 = non-fraud, 1 = fraud; highly imbalanced).
- **Preprocessing**: Standard scaling, SMOTE oversampling for training.

## Approach
1. **Data Loading & Exploration**:
   - Load dataset; view head, info, class distribution.
   - Visualize class imbalance with countplot.

2. **Preprocessing**:
   - Split into features (X) and target (y).
   - Train-test split (80/20); apply SMOTE to training data.
   - Standardize features.

3. **Modeling & Evaluation**:
   - Train models: Logistic Regression, Random Forest, XGBoost, Neural Network (with TensorFlow).
   - Evaluate: Accuracy, Precision, Recall, F1-Score, ROC AUC (e.g., Random Forest: ROC AUC 0.999999; XGBoost: 0.999995).
   - Select best model; visualize ROC curves and confusion matrix.

4. **Explainability**:
   - Use SHAP for feature importance in the best model.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `xgboost`, `tensorflow`, `shap`, `imblearn`

Install via:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost tensorflow shap imbalanced-learn
```

## Results
- Best performance: Random Forest (ROC AUC: 0.999999, Accuracy: 0.999868).
- Other models: XGBoost (ROC AUC: 0.999995), Neural Network (0.999933), Logistic Regression (0.997513).
- Visuals: ROC curves, confusion matrix.

## Notes
- Designed for Google Colab (Drive mount); adjust paths for local use.
- Handles imbalance; potential for hyperparameter tuning.

