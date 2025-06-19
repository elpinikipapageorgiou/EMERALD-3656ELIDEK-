# SPN CAD Codes: AI-Driven Clinical Classification for CAD and NSCLC

This repository contains a collection of Python scripts designed for the classification and evaluation of clinical data related to **Coronary Artery Disease (CAD)** and **Non-Small Cell Lung Cancer (NSCLC)**. The project integrates both structured clinical features and advanced AI models to benchmark predictive performance and interpretability.

## Key Features
- **Disease Focus**: CAD and NSCLC, including Solitary Pulmonary Nodules (SPNs)
- **Modeling Techniques**: Gradient Boosting (XGBoost), clinical rule-based logic, and statistical modeling
- **Evaluation Tools**:
  - Reliability Diagrams
  - KS (Kolmogorov–Smirnov) Statistic
  - Feature Importance Plots
  - Cohen’s Kappa Score

## File Overview
- `spn_main.py`: Main execution script for SPN classification models
- `spn_main_CAD.py`: CAD-specific model pipeline
- `spn_main_tune_xgb_noplots.py`: Hyperparameter tuning for XGBoost models (without plotting)
- `spn_model_evaluation_plots.py`: Generates comparison plots between predicted and true labels
- `spn_clinical_models.py`: Contains clinical prediction models and thresholds
- `spn_clinical_functions.py`: Utility functions for preprocessing, scoring, and formatting
- `main.py`: Legacy script for model comparison or experimentation

## 📊 Evaluation Metrics
- **Accuracy & F1-score**
- **ROC-AUC**
- **Kappa Statistic**
- **KS Statistic**
- **Feature Attribution**

## Purpose
To provide reproducible, interpretable benchmarks for comparing machine learning predictions against expert clinical labels in high-stakes diagnostic scenarios.

## Audience
- Clinical data scientists
- AI researchers in medical diagnostics
- Healthcare ML practitioners
