🧠 Machine Learning–Based Biometric Face Recognition & Authentication
(Decision Tree, Random Forest, AdaBoost, KNN, LDA)

This repository implements a complete biometric face recognition and authentication system using classical machine learning algorithms. It supports 1:1 verification, 1:N recognition, unknown-person detection, PCA-based feature extraction, threshold calibration, and automatic model persistence.

The code is fully based on the Python implementation in the file bfra_model.py.


bfra_model

📌 Key Features

Multi-model classifier training

Decision Tree

Random Forest

AdaBoost

K-Nearest Neighbors (KNN)

Linear Discriminant Analysis (LDA)

Biometric authentication modes
✔ 1:1 Face Verification (Cosine similarity)
✔ 1:N Identification (Best-match search)
✔ Unknown person detection with threshold score

Face Preprocessing

Grayscale conversion

64×64 resizing

Normalization

PCA transformation (Eigenfaces)

Automatic threshold calibration using ROC curve

Model Saving System

PCA saved as pca_model.pkl

Best ML model as best_model.pkl

Learned embeddings stored in HDF5 (embeddings.h5)

Threshold stored in threshold.json

Complete training pipeline + visualizations

📂 Dataset

The project uses the LFW Deep Funneled dataset.

Steps handled automatically:

Upload .zip

Extract to /dataset/lfw-deepfunneled

Filter persons with ≥ 2 images

Preprocess & flatten faces

🏗️ System Architecture

Below is the architecture that your script auto-generates as a Graphviz diagram:

Dataset → Preprocessing → PCA → ML Classifiers → Trained Model + Embeddings  
                        ↓                              ↑
                Input Image → Preprocess → PCA → Similarity Check → Result


Trained models + PCA + embeddings are then used for authentication.

🧬 Workflow
1. Preprocessing

Load images

Convert to grayscale

Resize to 64×64

Normalize and flatten

PCA (150-component Eigenfaces)

2. Classical ML Training

Models trained on PCA feature vectors:

Decision Tree

Random Forest

AdaBoost

KNN

LDA (best performer ≈ 27.6%)

3. Authentication & Recognition
✅ 1:1 Verification

Compare PCA embeddings using cosine similarity

If similarity ≥ threshold → Match

✅ 1:N Recognition

Compare input embedding with all stored person embeddings

Identify the best match

If score < threshold → Unknown Person

4. Threshold Calibration

Compute same-person vs different-person similarities

Generate ROC curve

Choose optimal threshold (TPR–FPR maximization)

📊 Final Model Performance
Model	Accuracy
LDA	0.276
Random Forest	0.086
KNN	0.074
AdaBoost	0.045
Decision Tree	0.023

(Values taken directly from the computed output.)


bfra_model

💾 Saved Model Format

Your script automatically saves:

File	Description
pca_model.pkl	PCA (Eigenfaces) model
best_model.pkl	Best ML model (Random Forest fallback)
threshold.json	Optimal similarity threshold
embeddings.h5	Compressed face embeddings per person

This allows real-time authentication without retraining.

🚀 Installation
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt


Libraries used:

OpenCV

NumPy

Scikit-learn

Scikit-image

Joblib

Matplotlib

h5py

Graphviz (optional for visualization)

▶️ How to Use
1. Train Models

Training is already included inside the script:

python bfra_model.py


This will:

load dataset

preprocess images

run PCA

train ML models

compute scores

generate embeddings

save model files

2. Authenticate a Face (Unified System)

Use the function:

authenticate_face("path/to/test.jpg")


Output example:

✅ Match Found: Bill_Gates (Similarity: 0.82)


Or for unknown:

❌ Unknown Person (Similarity: 0.42)

🖼️ Architecture Diagram

Your script auto-generates a high-quality Graphviz PNG file:

system_architecture_detailed.png

📜 MIT License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 Aniket

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

(Full MIT license text here... you can copy from MIT template)


<p align="center">

  <!-- Project Title -->
  <h2>🧠 Biometric Face Recognition & Authentication (ML + PCA)</h2>

  <!-- Main Badges -->
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-LFW%20(Kaggle)-blueviolet?style=for-the-badge" />

  <!-- ML Models -->
  <img src="https://img.shields.io/badge/Decision%20Tree-Classifier-forestgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Random%20Forest-Classifier-darkgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/AdaBoost-Classifier-red?style=flat-square" />
  <img src="https://img.shields.io/badge/KNN-Classifier-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/LDA-Linear%20Discriminant-purple?style=flat-square" />

  <!-- Libraries -->
  <img src="https://img.shields.io/badge/OpenCV-4.x-critical?style=flat-square&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellowgreen?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-Array%20Ops-informational?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-%23d97706?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-Model%20Saving-%23007ec6?style=flat-square" />
  <img src="https://img.shields.io/badge/HDF5-Embeddings-%23000099?style=flat-square" />

</p>

![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-LFW%20(Kaggle)-purple?style=for-the-badge)

<img src="https://img.shields.io/badge/Face%20Recognition-ML%20Pipeline-ff7b00?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PCA-Eigenfaces-6f42c1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/1:1%20Verification-Cosine%20Similarity-0fa3b1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/1:N%20Identification-Embeddings-3c096c?style=for-the-badge"/>

🙌 Acknowledgements

LFW Dataset

Scikit-learn community

OpenCV contributors
