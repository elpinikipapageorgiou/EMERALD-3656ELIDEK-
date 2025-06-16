
# 🧠 FCM-PSO Based Clinical Prediction and Explanation System

This project implements a **Fuzzy Cognitive Map (FCM)** model enhanced with a Particle Swarm Optimization **(PSO)** for predictive modeling and decision explanation in clinical datasets — specifically, for **Coronary Artery Disease (CAD)**.

It uses a trained FCM weight matrix to classify new patient cases, generate interpretability graphs, and provide **factual** and **counterfactual explanations** per instance.

---

## 📂 Project Structure

- `main.py` — Core script for FCM inference and visualization
- `dataset.xlsx` — Input clinical dataset for prediction
- `mean_values.xlsx` — Pre-trained FCM weight matrix (from PSO optimization)
- `fcm_graph_*.png` — Auto-generated causal graphs per patient

---

## 🔧 Setup

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn openpyxl matplotlib networkx tensorflow keras opencv-python
```
