# Hazardous_Asteroid_Classifier
🌌 Hazardous Asteroid Classifier 🚀

A Machine Learning Pipeline for Predicting Potentially Hazardous Asteroids & Detecting Anomalies

⭐ Overview

This project builds a complete end-to-end ML pipeline to classify asteroids as Hazardous or Non-Hazardous using NASA NEO dataset features.
It also performs anomaly detection using two different approaches — Isolation Forest and a custom Autoencoder.
SHAP Explainability is used to understand which features influence the model the most.

🛰️ Key Features
🔭 1. Data Cleaning & Feature Engineering

Conversion of approach year/month/day into a unified approach_date

Time-based features such as time_until_approach

Orbital mechanics features:

eccentricity

orbital period

specific angular momentum

velocity at perihelion & aphelion

Custom engineered features:

danger_score

approach_severity

energy_ratio

🧭 2. Anomaly Detection

Two independent methods were used:

📌 Isolation Forest (unsupervised)

Identifies anomalous asteroid behavior based on distance, velocity, and orbital metrics.
Outputs: Anomaly_IF

📌 Autoencoder (custom deep learning model)

Reconstructs asteroid feature vectors and flags high-loss samples as anomalies.
Outputs: anomaly_using_autoencoder

A confusion matrix compares agreement between both anomaly detectors.

🌋 3. Classification Model — XGBoost

The primary model predicts whether an asteroid is Hazardous (1) or Safe (0).

Included:

Train-test split

SMOTE oversampling

K-Fold Cross Validation (K = 2–10)

LogLoss & Accuracy plots for each K

Hyperparameter optimisation

🧠 4. Explainability — SHAP Values

SHAP plots highlight which engineered & physical features most strongly affect hazard classification.

📊 5. Visualizations

The notebook generates:

Pairplots for feature relationships

Correlation heatmap

ROC Curve

Confusion Matrix

SHAP summary bar plot

Anomaly distribution plots

🚀 Project Structure
Hazardous_Asteroid_Classifier/
│── data/                         # Dataset (not included in repo)
│── models/                       # Saved XGBoost + Autoencoder models
│── visuals/                      # All plots generated
│── Astroid_Detector_ML.ipynb     # Main notebook
│── requirements.txt              # Dependencies
│── README.md                     # Documentation

🧪 Tech Stack

Python 3.10

Pandas, NumPy, Seaborn, Matplotlib

TensorFlow / Keras

scikit-learn

SMOTE (imblearn)

XGBoost

SHAP

📈 Results Summary

XGBoost achieves strong accuracy on predicting hazardous asteroids

SHAP confirms importance of:

Risk Score

Miss Distance

Orbital Period

Velocity features

Isolation Forest and Autoencoder highlight potential unusual asteroid behaviors

🛠️ How to Run
1️⃣ Clone the repo
git clone https://github.com/PranavJain-GOAT/Hazardous_Asteroid_Classifier.git
cd Hazardous_Asteroid_Classifier

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook Astroid_Detector_ML.ipynb
