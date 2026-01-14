Credit Card Fraud Detection – Logistic Regression
## Project Overview

This project is an end-to-end Machine Learning classification system that detects fraudulent credit card transactions using Logistic Regression.
The model is trained on a highly imbalanced dataset and deployed as a REST API using FastAPI and Docker.

## Problem Statement

Credit card fraud is a rare but costly event.
The dataset contains thousands of normal transactions and very few fraud cases, making this a class-imbalance problem.

Goal:

Correctly identify fraud transactions

Minimize false negatives (missing fraud)

## Dataset

Dataset: Credit Card Transactions

Features:

Time

V1 to V28 (PCA-transformed features)

Amount

Target:

Class

0 → Normal transaction

1 → Fraud transaction

## Steps Followed (What I Did)
1️⃣ Data Understanding & Preprocessing

Checked for missing values (none found)

Identified severe class imbalance

Split data using stratified train-test split

Applied StandardScaler for feature scaling

2️⃣ Model Selection

Used Logistic Regression (binary classification)

Chosen because:

Interpretable

Supports probability output

Works well with imbalanced data

3️⃣ Handling Imbalanced Data

Used:

class_weight = "balanced"


Why?

Fraud cases are rare

This forces the model to give more importance to fraud samples

Prevents the model from blindly predicting “Normal” every time

4️⃣ Hyperparameter Tuning

Used GridSearchCV with:

C → Regularization strength

penalty → L2 regularization

solver → lbfgs

scoring → ROC-AUC

cv → 5-fold cross validation

Why ROC-AUC?

Accuracy is misleading for imbalanced datasets

ROC-AUC measures ranking quality of predictions

5️⃣ Threshold Tuning

Instead of default 0.5, used:

THRESHOLD = 0.2


Reason:

In fraud detection, missing fraud is worse than false alarm

Lower threshold → higher fraud recall

6️⃣ Model Evaluation

Used:

Confusion Matrix

Classification Report

ROC-AUC score

Probability distribution visualization

## Deployment 
🔹 Backend API

Built using FastAPI

Endpoint:

POST /predict


Example input:

{
  "features": [30 numerical values]
}


Example output:

{
  "fraud_probability": 0.34,
  "prediction": 1
}

🔹 Dockerization

Created Dockerfile

Packaged:

Trained model

Scaler

FastAPI app

Exposed API on port 8000

🐞 Errors Faced & How I Fixed Them
❌ Error 1: predict_proba not found

Cause:
Accidentally deployed a LinearRegression model instead of LogisticRegression.

Fix:
Re-saved the correct Logistic Regression model and rebuilt Docker image.

❌ Error 2: Feature mismatch

Cause:
Mismatch between training features and API input.

Fix:
Ensured same feature order and count in training and inference.

❌ Error 3: Docker port already in use

Cause:
Previous container still running.

Fix:
Stopped old container or used a different port.

## Key Learnings

Logistic Regression is classification, not regression

Accuracy is misleading for imbalanced datasets

class_weight and threshold tuning are critical

Deployment failures are often pipeline mismatch issues

Docker runs only what you copy into the image

## Tech Stack

Python

NumPy, Pandas

Scikit-learn

FastAPI

Docker

Uvicorn


## Final Note

This project simulates a real-world fraud detection system, covering ML + deployment + debugging, not just model training.# logisticregression
