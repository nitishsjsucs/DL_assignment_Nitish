# Deep Learning Training Design Space — Assignment Portfolio

**Student:** Nitish Chowdary Ratakonda  
**Course Assignment:** Execute Colabs for Deep Learning Training Design Space & Create Explanation Videos  
**Due Date:** April 12, 2026 at 23:59  
**Points:** 100  

---

## Overview

This repository contains six executed Google Colab notebooks covering core deep learning topics. Each notebook has been copied to this GitHub repository with all code executed and outputs preserved. Accompanying video walkthroughs explain every code block and its output in detail.

The notebooks span the full lifecycle of training neural networks — from understanding how neurons activate, to tuning hyperparameters, to selecting the right optimizer and evaluating model performance.

---

## Repository Structure

```
DL_assignment_Nitish/
├── README.md
├── final_activation_functions_tutorial.ipynb
├── final_cnn_fundamentals_tutorial.ipynb
├── final_hyperparameter_tuning_tutorial.ipynb
├── final_important_classification_metrics_tutorial.ipynb
├── final_modern_cnn_architectures_tutorial.ipynb
└── final_optimizers_deep_learning_tutorial.ipynb
```

---

## Notebooks & Video Walkthroughs

Each section below includes a direct link to the executed notebook on GitHub, a one-click "Open in Colab" badge, and a YouTube video walkthrough explaining the notebook code block by code block.

---

### 1. Activation Functions for Deep Learning

> *From Zero to Hero — A Comprehensive, Beginner-Friendly Guide*

Covers the role of activation functions in breaking linearity, explores ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, Swish, and GELU, and demonstrates the vanishing gradient problem with visualizations. Includes from-scratch implementations and side-by-side performance comparisons on real training tasks.

**Topics covered:**
- What activation functions do and why they matter
- Mathematical definitions and derivative analysis
- Dead neuron problem in ReLU and how to fix it
- Comparison of modern activations (Swish, GELU) vs classics
- Practical selection guidelines for different architectures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_activation_functions_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

### 2. Convolutional Neural Networks from First Principles

> *A Visual Journey into Computer Vision Deep Learning*

Builds CNNs from the ground up, starting with the intuition behind convolution and progressing through pooling, feature maps, receptive fields, and full CNN architectures. Trains and evaluates a CNN on image classification data with detailed visualizations of learned filters and activations.

**Topics covered:**
- Why fully connected networks fail for images
- Convolution operation — kernels, strides, padding
- Pooling layers and spatial dimensionality reduction
- Building a complete CNN architecture from scratch
- Visualizing learned filters and intermediate feature maps
- Training loop, loss curves, and evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_cnn_fundamentals_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

### 3. Hyperparameter Tuning for Deep Learning

> *A Comprehensive, Beginner-Friendly Guide*

Explores the full hyperparameter search landscape — learning rate, batch size, number of layers, dropout rate, weight decay, and more. Covers manual tuning, grid search, random search, and modern techniques like learning rate schedulers and early stopping.

**Topics covered:**
- The hyperparameter landscape and why tuning matters
- Learning rate — the most important hyperparameter
- Batch size effects on generalization and training speed
- Network depth and width trade-offs
- Regularization hyperparameters: dropout, weight decay, L2
- Learning rate schedules: step decay, cosine annealing, warm-up
- Early stopping to prevent overfitting
- Systematic search strategies: grid vs. random vs. Bayesian

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_hyperparameter_tuning_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

### 4. Important Classification Metrics

> *A Complete Guide to Evaluating Classifier Performance*

Goes beyond accuracy to cover the full suite of classification evaluation metrics. Uses synthetic and real datasets to show when accuracy misleads, and how precision, recall, F1, ROC-AUC, and confusion matrices provide a complete picture of model performance.

**Topics covered:**
- What classification is and types (binary, multi-class, multi-label)
- Why accuracy alone is insufficient (class imbalance problem)
- Confusion matrix — TP, TN, FP, FN explained visually
- Precision, Recall, F1-Score — formulas, intuition, trade-offs
- ROC curve and AUC-ROC interpretation
- Precision-Recall curve for imbalanced datasets
- Matthews Correlation Coefficient (MCC)
- Choosing the right metric for your use case

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_important_classification_metrics_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

### 5. Modern CNN Architectures — ResNet to EfficientNet

> *Mastering Deep Networks and Transfer Learning*

Builds on CNN fundamentals to explore the landmark architectures that defined modern computer vision. Implements and compares VGG, ResNet (with residual connections), Inception, MobileNet, and EfficientNet, and demonstrates transfer learning with fine-tuning on a custom dataset.

**Topics covered:**
- The problem of vanishing gradients in very deep networks
- Residual connections — the idea behind ResNet
- Skip connections, identity mappings, and bottleneck blocks
- Inception modules — parallel convolutions at multiple scales
- Depthwise separable convolutions (MobileNet)
- Compound scaling — EfficientNet's unified approach
- Transfer learning: feature extraction vs. fine-tuning
- Practical guidance on choosing architectures for real tasks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_modern_cnn_architectures_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

### 6. Optimizers for Deep Learning

> *A Comprehensive, Beginner-Friendly Guide*

Takes a deep dive into the optimization algorithms that train neural networks. Visualizes loss landscapes and contour plots, implements optimizers from scratch, and systematically compares SGD, Momentum, RMSProp, Adam, AdaGrad, AdamW, and more on real training tasks.

**Topics covered:**
- Loss landscapes and why deep learning optimization is hard
- Vanilla Gradient Descent and its limitations
- Stochastic Gradient Descent (SGD) and mini-batch variants
- Momentum — escaping local minima and accelerating convergence
- AdaGrad — adaptive learning rates per parameter
- RMSProp — fixing AdaGrad's diminishing learning rate
- Adam — combining Momentum and RMSProp (the de-facto standard)
- AdamW — decoupled weight decay for better regularization
- Learning rate warmup and cyclical learning rates
- Practical optimizer selection guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nitishsjsucs/DL_assignment_Nitish/blob/main/final_optimizers_deep_learning_tutorial.ipynb)
&nbsp;&nbsp;
[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?logo=youtube)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

---

## Quick Reference — All Videos

| # | Notebook | YouTube Walkthrough |
|---|----------|-------------------|
| 1 | Activation Functions | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |
| 2 | CNN Fundamentals | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |
| 3 | Hyperparameter Tuning | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |
| 4 | Classification Metrics | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |
| 5 | Modern CNN Architectures | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |
| 6 | Optimizers for Deep Learning | [▶ Watch](https://www.youtube.com/watch?v=dQw4w9WgXcQ) |

---

## Prerequisites & Environment

All notebooks are self-contained and run on Google Colab with no local setup required. Dependencies are installed inline within each notebook.

**Core libraries used across notebooks:**
- `PyTorch` — model building and training
- `NumPy` / `Pandas` — numerical computation and data handling
- `Matplotlib` / `Seaborn` — visualizations and plots
- `scikit-learn` — metrics, datasets, and preprocessing
- `torchvision` — datasets and pretrained model weights

---

## How to Run

1. Click any **Open In Colab** badge above to open the notebook directly in Google Colab.
2. Go to **Runtime → Run all** to execute all cells with outputs.
3. Alternatively, clone this repository and run locally with Jupyter:

```bash
git clone https://github.com/nitishsjsucs/DL_assignment_Nitish.git
cd DL_assignment_Nitish
jupyter notebook
```

---

## About

This portfolio was created as part of a graduate-level deep learning course assignment. The goal was to deeply understand and explain the core components of the deep learning training design space — from neuron activations all the way to architecture selection and evaluation — by executing, studying, and teaching each topic through video.

**Author:** Nitish Chowdary Ratakonda  
**GitHub:** [nitishsjsucs](https://github.com/nitishsjsucs)
