# 🧠 Machine Learning–Based Biometric Face Recognition & Authentication (Decision Tree, Random Forest, AdaBoost, KNN, LDA)

This repository implements a complete biometric face recognition and authentication system using classical machine learning algorithms. It supports 1:1 verification, 1:N identification, unknown-person detection, PCA-based feature extraction, and threshold calibration.

The complete implementation is inside BFRA_model.ipynb, which contains preprocessing, model training, authentication logic, embedding generation, ROC-based threshold tuning, and model saving.

bfra_model

 # 📦 Repository Structure

Your repo folder structure (from your screenshot):

📁 dataset/             → LFW People dataset (Kaggle link included below)
📁 saved_models/        → Saved PCA, ML models, embeddings, threshold.json
📁 venv/                → Local virtual environment (optional to commit)
📄 BFRA_model.ipynb     → Full training, PCA, authentication & recognition pipeline
📄 requirements.txt     → Required Python dependencies

# 📥 Dataset Source (Kaggle)

This project uses the LFW People (Labeled Faces in the Wild) dataset:

👉 https://www.kaggle.com/datasets/atulanandjha/lfwpeople

How to use:

Download the dataset ZIP from Kaggle

Extract it to:

dataset/lfw-deepfunneled/


Ensure the structure is:

dataset/
 └── lfw-deepfunneled/
      ├── Person1/
      ├── Person2/
      └── ...

# 🚀 Features
🖼️ Preprocessing

Grayscale conversion

64×64 face resizing

Normalization

PCA whitening (Eigenfaces, 150 components)

🤖 ML Models Implemented

Decision Tree

Random Forest

AdaBoost

KNN

LDA (Best performance in this dataset)

🔐 Authentication Modes
1. 1:1 Verification

Compares two faces using cosine similarity
→ If similarity ≥ threshold → Match

2. 1:N Identification

Compares user input against all stored embeddings
→ Best match returned
→ If score < threshold → Unknown Person

📊 Threshold Calibration

ROC curve computation

Optimal threshold selection (TPR − FPR maximization)

💾 Model Saving (Auto-Save)

Your notebook stores:

File	Located in	Purpose
pca_model.pkl	saved_models/	PCA eigenface model
best_model.pkl	saved_models/	Best ML classifier (fallback RF)
embeddings.h5	saved_models/	Serialized face embeddings
threshold.json	saved_models/	Optimal similarity threshold

# 📊 Model Accuracy Summary
Model	Accuracy
LDA	0.276
Random Forest	0.086
KNN	0.074
AdaBoost	0.045
Decision Tree	0.023

(Extracted from the notebook output)


bfra_model

# ⚙️ Installation
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt


(requirements.txt matches your environment)

# ▶️ Running the System
Open the main file:

BFRA_model.ipynb

Run the notebook cells in order:

✔ Loads dataset
✔ Preprocesses faces
✔ Extracts PCA features
✔ Trains all ML models
✔ Compares their accuracy
✔ Generates embeddings
✔ Saves PCA + models + embeddings
✔ Provides authentication API

# 🔐 Authentication Function
authenticate_face("dataset/lfw-deepfunneled/Bill_Gates/Bill_Gates_0003.jpg")


Output:

✅ Match Found: Bill_Gates (Similarity: 0.82)

Or:

❌ Unknown Person (Similarity: 0.42)


# 🏗️ System Architecture

Your notebook auto-generates a Graphviz PNG diagram:

system_architecture_detailed.png


Pipeline:

Dataset → Preprocessing → PCA → ML Classifiers → Saved Model & Embeddings
                       ↓                                 ↑
                Input Image → Preprocess → PCA → Similarity → Result

# 📜 License – MIT

Your project is released under the MIT License.

MIT License

Copyright (c) 2025 Aniket

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files…

(Include the full MIT license text here)
