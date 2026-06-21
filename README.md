# Breast Cancer Classification Using Deep Learning Feature Extraction

A machine learning project for **COMP 381 - Introduction to Machine Learning** at the **University of the Fraser Valley**.

This project classifies breast ultrasound images as **benign** or **malignant** using transfer learning for feature extraction combined with traditional supervised and unsupervised ML classifiers.

---

## Team

| Name | GitHub |
|------|--------|
| Charan Elam | [@CharanElam](https://github.com/CharanElam) |
| Diana Emal | [@dianaemal](https://github.com/dianaemal) |
| Sana Elhag | [@SanaElhag](https://github.com/SanaElhag) |

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Dependencies](#dependencies)

---

## Overview

Breast cancer is one of the most common cancers worldwide, and early detection significantly improves patient outcomes. This project explores whether classic ML classifiers — trained on deep features extracted from a pretrained CNN — can accurately distinguish benign from malignant breast ultrasound images.

**Key techniques used:**
- Transfer learning with **VGG16** (supervised) and **ResNet50** (unsupervised) pretrained on ImageNet
- Dimensionality reduction with **PCA** (70 components) to address overfitting from high-dimensional features (25,088)
- Feature standardization with **StandardScaler**
- Supervised classifiers: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- Unsupervised clustering: **K-Means** (k=2, k=3)

---

## Dataset

The dataset contains ~647 breast ultrasound images split into two classes:

| Class | Count |
|-------|-------|
| Benign | 437 |
| Malignant | 210 |

Images vary in size (e.g., 685×567, 792×571 pixels) and are stored as PNG files.

**To run this project you must download the dataset separately:**

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1VtW8dB1ru4QY1O1aw2skS9j_eg44t-zR?usp=sharing).
2. Place the downloaded `raw` folder (which contains `benign/` and `malignant/` subfolders) inside the `data/` directory.
3. Do **not** push this folder — it is excluded via `.gitignore`.

---

## Project Structure

```
breast-cancer-analysis/
├── data/
│   ├── raw/                    # Downloaded dataset (not tracked in git)
│   │   ├── benign/
│   │   ├── malignant/
│   │   ├── benign_mask/
│   │   └── malignant_mask/
│   └── features/               # Extracted features saved as .npy arrays
│       ├── X_features.npy
│       └── y_labels.npy
├── notebooks/
│   ├── 01-data-exploration.ipynb     # Dataset inspection and image stats
│   ├── 02-data-processing.ipynb      # VGG16 feature extraction & saving
│   ├── 03-model-training.ipynb       # Supervised ML models (LogReg, KNN, SVM)
│   └── 04-unsupervised-model.ipynb   # K-Means clustering with ResNet50
├── results/
│   ├── Results.ipynb                 # Summary notebook with all results
│   ├── model_results.csv             # Accuracy, precision, recall, F1, AUC
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── comparisons.png
└── README.md
```

---

## Methodology

### 1. Data Exploration (`01-data-exploration.ipynb`)

Inspected the dataset structure, counted images per class, and examined image dimensions. Found 437 benign and 210 malignant images with varying resolutions.

### 2. Feature Extraction (`02-data-processing.ipynb`)

Used **VGG16** (pretrained on ImageNet, top layer removed) to extract spatial features from each image:
- Images resized to **224×224** (VGG16's expected input size)
- Features extracted per image: **7×7×512 = 25,088 values**
- All 647 feature vectors and labels shuffled and saved as `.npy` arrays

### 3. Supervised Model Training (`03-model-training.ipynb`)

**Preprocessing:**
- Flattened features from `(647, 1, 7, 7, 512)` → `(647, 25088)`
- 80/20 train-test split (517 train / 130 test)
- Applied `StandardScaler` then **PCA with 70 components** to reduce overfitting (raw features caused ~99.8% train accuracy)

**Models trained:**

| Model | Notes |
|-------|-------|
| Logistic Regression (default threshold 0.5) | L2 regularization, C=0.01, class_weight='balanced' |
| Logistic Regression (threshold 0.3) | Lower threshold to reduce false negatives (missed cancers) |
| K-Nearest Neighbors | k=7 |
| Support Vector Machine | RBF kernel, C=10, class_weight='balanced' |

### 4. Unsupervised Clustering (`04-unsupervised-model.ipynb`)

Used **ResNet50** (pretrained on ImageNet, global average pooling) to extract a 2048-dim feature vector per image. Applied **K-Means clustering** with k=2 and k=3, then visualized cluster separation using PCA (2 components).

---

## Results

All models were evaluated on the held-out test set (130 images).

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression (t=0.5) | **89.2%** | 0.800 | 0.800 | 0.800 | **0.863** |
| Logistic Regression (t=0.3) | 83.8% | 0.646 | **0.886** | 0.747 | 0.853 |
| KNN (k=7) | 84.6% | 0.727 | 0.686 | 0.706 | 0.795 |
| SVM (RBF) | **89.2%** | **0.818** | 0.771 | 0.794 | 0.854 |

**Key observations:**
- Logistic Regression and SVM tied for highest accuracy (89.2%) and AUC.
- Lowering the Logistic Regression threshold to 0.3 increased recall for malignant cases (0.80 → 0.89) at the cost of precision — useful in a clinical context where missing a cancer is more costly than a false alarm.
- KNN underperformed relative to the other models in this feature space.

Result visualizations (confusion matrices, ROC curves, model comparisons) are available in the [`results/`](results/) folder.

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/CharanElam/breast-cancer-analysis.git
cd breast-cancer-analysis
pip install tensorflow scikit-learn numpy matplotlib pandas pillow
```

### Running the Notebooks

Run the notebooks in order:

```
1. notebooks/01-data-exploration.ipynb   — explore the dataset
2. notebooks/02-data-processing.ipynb   — extract and save VGG16 features
3. notebooks/03-model-training.ipynb    — train and evaluate supervised models
4. notebooks/04-unsupervised-model.ipynb — run K-Means clustering
```

> **Note:** Notebook 02 must be run before 03 to generate the `.npy` feature files in `data/features/`. This step requires the raw dataset to be downloaded first (see [Dataset](#dataset)).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` / `keras` | VGG16 & ResNet50 feature extraction |
| `scikit-learn` | ML models, PCA, StandardScaler, metrics |
| `numpy` | Array operations and feature storage |
| `matplotlib` | Plotting results |
| `pandas` | Results tabulation |
| `Pillow` | Image loading |

---

*University of the Fraser Valley — COMP 381: Introduction to Machine Learning*
