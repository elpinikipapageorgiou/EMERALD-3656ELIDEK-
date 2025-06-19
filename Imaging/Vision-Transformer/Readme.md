# Vision Transformer (ViT) for Image Classification

This repository implements a Vision Transformer (ViT)-based image classification model using TensorFlow and Keras, inspired by the official [Keras Vision Transformer example](https://keras.io/examples/vision/image_classification_with_vision_transformer/).

---

## Description

This model applies Transformer-based learning to image classification tasks by:
- Splitting images into fixed-size patches
- Projecting them into a lower-dimensional embedding space
- Passing them through multiple Transformer layers
- Applying a Multi-Layer Perceptron (MLP) for classification

It supports multi-class classification and uses modern practices like `AdamW` optimization and dropout regularization.

---

## 📦 Features

- ✅ Patch extraction and positional encoding
- ✅ Custom Transformer layers with multi-head attention
- ✅ Fully configurable depth, dimensions, and MLP head
- ✅ Sparse categorical crossentropy loss
- ✅ Top-1 and Top-5 accuracy metrics

---

## 🧪 Requirements

Install dependencies:

```bash
# Option 1: Conda
conda install -c esri tensorflow-addons

# Option 2: Pip
pip install tensorflow tensorflow-addons
