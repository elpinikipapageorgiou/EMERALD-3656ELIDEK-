# ML-NSCLC Classification (EMERALD-3656ELIDEK Multimodal Pipeline)

This repository contains a comprehensive deep learning and explainable AI framework for the classification of Non-Small Cell Lung Cancer (NSCLC) using multimodal data. It integrates clinical and imaging features (CT, PET, etc.) with state-of-the-art vision models and interpretability tools to support explainable diagnosis models.

---

## 📂 Structure Overview

| Category | Description |
|----------|-------------|
| `fr_ioapi_*.py` | Pretrained transformer backbones and CNN variants from ImageNet (ConvNeXt, FNet, ViT, Swin, etc.) |
| `spn_clinical_models.py` | Clinical-only models using tabular features |
| `spn_data_loader.py` | Multimodal data loader (clinical + imaging support) |
| `spn_model_maker.py` | Model construction logic for multimodal fusion |
| `spn_gradcam_func.py` / `spn_gradcamplusplus.py` | Grad-CAM and Grad-CAM++ visual explanation modules |
| `spn_lime_func.py` | LIME-based interpretability for tabular or image features |
| `spn_main.py` | Main training and evaluation loop |
| `spn_main_functions.py` | Utilities for model training, early stopping, logging |
| `spn_metrics.py` | Evaluation metrics (AUC, F1, precision, etc.) |
| `spn_feature_map.py` | Extracts and processes CNN feature maps |
| `spn_ml_model_evaluation_plots.py` | Confusion matrices, ROC, PR curves |
| `spn_main_feature_importance_calc.py` | SHAP / statistical importance ranking |
| `spn_model_evaluation_plots.py` | Visualization for clinical metrics |
| `spn_clinical_functions.py` | Preprocessing and encoding of clinical variables |

---

## 🚀 Key Features

- **Multimodal classification** using clinical + imaging data
- **NSCLC diagnosis support** with interpretable outputs
- Integrated **Grad-CAM**, **Grad-CAM++**, and **LIME** for XAI
- Multiple vision backbones (ConvNeXt, ViT, Swin, FNet, etc.)
- Evaluation: ROC, PR, confusion matrix, AUC, F1
- Feature importance via SHAP/statistical methods
- Supports ablation studies on unimodal vs multimodal input

