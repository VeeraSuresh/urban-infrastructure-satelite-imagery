# Extracting Urban Infrastructure Information Through Satellite Images

## Overview

This project presents a machine learning framework for extracting urban infrastructure information from high-resolution satellite imagery. The objective is to automatically classify road and building pixels using feature engineering and supervised machine learning techniques.

The solution combines image preprocessing, statistical analysis, feature extraction, dimensionality reduction, and classification to provide an interpretable and scalable approach for urban infrastructure mapping.

## Problem Statement

Rapid urbanisation has increased the need for accurate and up-to-date information about urban infrastructure. Traditional mapping methods based on manual surveys are expensive, time-consuming, and difficult to scale.

This project addresses the challenge by using satellite imagery and machine learning to automatically identify:

* Roads
* Buildings

from urban satellite images at pixel level.

---

## Dataset

The dataset consists of 203 high-resolution RGB satellite image tiles collected from:

* Mohammed Bin Rashid Space Centre (MBRSC)
* ISPRS Urban Segmentation Dataset
* Bhuvan Satellite Imagery

The images cover multiple Asian cities including:

* Shanghai
* Hangzhou
* Wuhan
* Chengdu
* Guangzhou

Each image includes pixel-level annotated masks used for supervised learning.

---

## Methodology

### 1. Data Preprocessing

* RGB normalisation
* Noise reduction
* Contrast enhancement
* Class balancing
* Training and testing split

### 2. Feature Engineering

Extracted features include:

#### Spectral Features

* RGB Channels
* Grayscale Intensity
* Excess Green (ExG)

#### Texture Features

* Local Variance
* Local Binary Patterns (LBP)

#### Structural Features

* Gradient Magnitude
* Laplacian Magnitude
* Edge Density
* Hough Line Features

#### Morphological Features

* Morphological Gradient
* Top-Hat Transform

#### Spatial Features

* Normalised X Position
* Normalised Y Position

### 3. Statistical Analysis

* Descriptive Statistics
* Feature Distribution Analysis
* Correlation Analysis
* Hypothesis Testing
* Confidence Intervals

### 4. Dimensionality Reduction

* Principal Component Analysis (PCA)
* Retained approximately 95% of feature variance

### 5. Machine Learning Models

The following classifiers were evaluated:

* K-Nearest Neighbours (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

## Results

### Classification Accuracy

| Model         | Accuracy |
| ------------- | -------- |
| KNN           | 81.46%   |
| Decision Tree | 74.62%   |
| Random Forest | 74.62%   |
| SVM           | 73.85%   |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Intersection over Union (IoU)

### Key Findings

* KNN achieved the highest classification accuracy.
* Random Forest provided the best balance between robustness and spatial consistency.
* Feature engineering significantly improved class separability.
* Statistical analysis confirmed meaningful differences between road and building classes.
* The framework successfully extracted urban infrastructure from diverse urban environments.

---

## Technologies Used

* MATLAB
* Image Processing Toolbox
* Statistics and Machine Learning Toolbox
* Principal Component Analysis (PCA)
* Random Forest
* Support Vector Machine
* K-Nearest Neighbours

---

## Project Structure

```text
project/
│
├── data/
│   ├── images/
│   └── masks/
│
├── src/
│   ├── preprocessing.m
│   ├── feature_extraction.m
│   ├── pca_analysis.m
│   ├── train_models.m
│   └── evaluate_models.m
│
├── results/
│   ├── confusion_matrix.png
│   ├── correlation_matrix.png
│   ├── segmentation_results.png
│   └── dashboard.png
│
├── report/
│   └── FinalReport.pdf
│
└── README.md
```

---

## Applications

This framework can support:

* Smart City Development
* Urban Planning
* Transportation Analysis
* Infrastructure Monitoring
* Land Use Analysis
* Environmental Monitoring
* Disaster Response Planning

---

## Future Enhancements

* Deep Learning-based Semantic Segmentation
* Multi-class Urban Feature Extraction
* Multi-temporal Satellite Analysis
* Higher Resolution Imagery Support
* Hybrid Machine Learning and Deep Learning Models

---

## Author

**Veera Suresh Akuthota**

MSc Data Science
University of Roehampton, London

---

## License

This project is intended for academic and research purposes.
