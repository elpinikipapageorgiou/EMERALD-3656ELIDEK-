# FCMExpNLG - Application to NSCLC

# NSCLC Diagnosis Using Fuzzy Cognitive Maps (FCMs)

This project implements an **explainable AI model** for **non-small cell lung cancer (NSCLC)** diagnosis using **Fuzzy Cognitive Maps (FCMs)**. It performs classification of patient data and provides **interpretable graph-based and textual explanations**.

---

## Key Features

- **NSCLC Classification**: Predicts benign vs malignant outcomes based on clinical and imaging-derived features.
- **FCM Visualization**: Generates detailed concept influence graphs (`fcm_graph_#.png`) showing weighted causal paths.
- **Textual Explanation**: Provides factual, human-readable reasoning for each classification (benign/malignant).
- **Input-Aware Inference**: Dynamically adjusts concept weights based on available patient features.
- **Custom Prediction Logic**: Sigmoid-based influence aggregation using a learned weight matrix.

---

## 🧠 Model Overview

- **Architecture**: Fuzzy Cognitive Map (FCM)
- **Input**: Preprocessed features (age, SUV, location, type, margins, etc.)
- **Output**: Binary prediction (`malignant` or `benign`)
- **Explanation**: Graph of influential concepts + tailored text explanation

---

## 🗂️ Files & Structure

| File                     | Description |
|--------------------------|-------------|
| `dataset.xlsx`           | NSCLC patient data with features and ground-truth labels |
| `mean_values.xlsx`       | Pre-trained FCM weight matrix for inference |
| `fcm_graph_#.png`        | Graphical visualization for each patient |
| `main.py`                | Full pipeline: load data → predict → explain → visualize |

---

## 📈 Output Example

```text
***************
Patient 3:
The patient is likely to have NSCLC

Factual Explanation:
Based on the input values provided for the patient case, the Fuzzy Cognitive Map (FCM) has determined the classification as malignant, primarily due to key clinical risk factors indicating a higher likelihood of cancer or disease progression.

The most significant contributing factors for malignancy include:
• High SUV: A higher Standardized Uptake Value (SUV) in PET scans may indicate increased metabolic activity...
• Spiculated margins: Spiculated margins are a common characteristic of malignant tumors...

# Usage

FCMExpNLG provides textual explanations to justify the decision making process of FCMs.

The dataset should be initialized here:
```
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')
```
# Supervisor

[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)

# Contributors

[Anna Feleki](https://emerald.uth.gr/personnel/)
[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)
[Ioannis Apostolopoulos](https://emerald.uth.gr/personnel/)
[Nikolaos Papandrianos](https://emerald.uth.gr/personnel/)
[Serafeim Moustakidis](https://emerald.uth.gr/personnel/)
