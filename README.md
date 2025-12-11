🚀 Hazardous Asteroid Classifier — ML Project

A complete Machine Learning pipeline for identifying potentially hazardous asteroids using NASA orbital and approach data.

📌 Project Overview

This project builds a robust machine learning classifier to categorize asteroids as Hazardous (1) or Not Hazardous (0) based on their physical and orbital characteristics.
It includes:

Complete EDA

Feature engineering based on orbital mechanics

Handling class imbalance

Multiple ML models with XGBoost as the final choice

K-Fold Cross Validation (K = 2 to 10)

Hyperparameter optimization

ROC & Confusion Matrix evaluation

SHAP explainability

Anomaly detection using Isolation Forest + Autoencoder

This README follows the exact workflow used inside the notebook.

🧭 1. Exploratory Data Analysis (EDA)
1.1 Data Inspection

Loaded dataset and checked shape, dtypes, missing values

Displayed head/tail samples

Identified numerical & categorical features

1.2 Statistical Distribution

Histograms for all continuous features

Checked skewness

Decided which features require scaling

1.3 Outlier Detection

Boxplots for each numerical feature

Z-score based outlier detection

Identified extreme values in:

Velocity

Miss Distance

Semi-Major Axis

1.4 Correlation Analysis

Heatmap showing feature correlations

Highlighted:

Semi-Major Axis ↔ Orbital Period (Kepler’s Law)

High correlation between distance measures

🛠️ 2. Feature Engineering
2.1 Date Features

📌 Combined approach_year, approach_month, approach_day into approach_date
📌 Extracted day_of_year

2.2 Physics-Inspired Generated Features

Added scientifically meaningful features:

Miss-to-Axis Ratio

Time Until Approach

Orbital Eccentricity

Average Orbital Velocity

Specific Orbital Energy

Escape Velocity

Specific Angular Momentum

Velocity at Perihelion & Aphelion

Synodic Period & Mean Motion

2.3 Additional Custom Features

These improve model separability:

📌 danger_score → ratio of velocity to miss distance
📌 approach_severity → inverse of time remaining
📌 energy_ratio → orbital energy / semi-major axis

⚖️ 3. Handling Class Imbalance

The dataset is highly imbalanced (Hazardous ≪ Not Hazardous).
Techniques applied:

SMOTE oversampling

Class weights during model training

Ensured balanced splits using stratify=y

🏗️ 4. Model Building

Multiple models were tested:

Model	Result
Logistic Regression	Baseline
Random Forest	Good but unstable
XGBoost ⭐	Best results
LightGBM	Optional alternative

XGBoost was selected as the final model due to:

Best ROC-AUC

Handles imbalance

Works well with engineered features

Interpretable via SHAP

🔄 5. K-Fold Cross Validation (K = 2–10)

Implemented training loops for K = 2 to 10.
For each K:

Performed SMOTE inside each fold

Collected logloss and accuracy curves

Plotted:

Validation Loss vs Epochs

Validation Accuracy vs Epochs

Outcome:

✔ Model stable across folds
✔ Lower K overfits slightly
✔ Higher K generalizes better

🎯 6. Hyperparameter Optimization

Used RandomizedSearchCV / manual grid tuning:

max_depth

learning_rate

subsample

colsample_bytree

n_estimators

Best model saved and evaluated.

📈 7. Model Evaluation

Metrics computed:

Accuracy

Precision

Recall (important for Hazardous class)

F1-Score

Confusion Matrix

ROC Curve (AUC)

🔍 8. Anomaly Detection

Two anomaly detection frameworks:

8.1 Isolation Forest (in-built method)

Trained on numerical features

Labeled anomalies as Anomaly_IF

8.2 Autoencoder (custom deep learning method)

Built a symmetric encoder–decoder NN

Trained to reconstruct normal samples

Used reconstruction error to detect anomalies

Added Anomaly_AE column

8.3 Comparison

Confusion Matrix between methods

Count of anomalies detected by both

Intersection of anomaly sets

🧠 9. Explainability — SHAP

Used SHAP GradientExplainer / Explainer (since TreeExplainer had compatibility issues with XGBoost).

Visualizations include:

SHAP bar plot (feature importance)

SHAP summary plot (feature impact)

Key Insights:

Miss Distance features were the strongest predictors

Velocity and orbital energy also played major roles

📦 10. Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

XGBoost

Scikit-Learn

TensorFlow/Keras

SHAP

Imbalanced-Learn

Isolation Forest

Autoencoders


