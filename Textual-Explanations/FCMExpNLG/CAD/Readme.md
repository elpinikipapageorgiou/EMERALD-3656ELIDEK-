# FCMExpNLG

# FCM-Based Clinical Decision Support System for CAD Diagnosis

This project implements a clinical decision support system (CDSS) based on **Fuzzy Cognitive Maps (FCMs)** for the classification and explanation of Coronary Artery Disease (CAD). It supports per-patient prediction, visualization of concept influence graphs, and generates personalized factual explanations.

---

## Key Features

- **FCM Visualization**: Generates concept graphs with weighted arrows showing influence.
- **Dynamic Inference**: Makes predictions using a sigmoid-based FCM matrix multiplication.
- **Input Sanitization**: Handles missing inputs and adapts the concept weights accordingly.
- **Factual Explanation Generator**: Provides human-readable clinical explanations for the output.
- **Concept Strength Mapping**: Assigns verbal labels (e.g., "Strong", "Very Weak") to concept influences.

---

## Model Description

- **Model Type**: Fuzzy Cognitive Map (FCM)
- **Diagnosis Goal**: Classify each patient as "normal" or "pathological" regarding CAD likelihood.
- **Inputs**: Clinical features such as age, diabetes, hypertension, smoking, previous interventions, etc.
- **Output**: A binary prediction (`normal` or `pathological`) with supporting explanation and concept graph.

---

## 🗂️ Folder & File Structure

| File                         | Description |
|------------------------------|-------------|
| `dataset.xlsx`               | Input patient dataset (rows = patients, columns = clinical features). |
| `mean_values.xlsx`           | Final trained FCM weight matrix (used for inference). |
| `fcm_graph_#.png`            | Output visualizations showing weighted concept influence per patient. |
| `main.py` (this script)      | Full prediction, explanation, and graph generation pipeline. |

---

## 📊 Output

For each patient:
- A **prediction** is made (`normal` or `pathological`)
- A **graph** is saved as `fcm_graph_#.png` showing the influence of each input concept.
- A **textual explanation** is generated with medical reasoning behind the decision.

---

## 🧾 Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `openpyxl`
- `networkx`
- (Optional) `mpl_toolkits.axes_grid1` for advanced graph visualization

Install all dependencies:
```bash
pip install numpy pandas matplotlib openpyxl networkx


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
