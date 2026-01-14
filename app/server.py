from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("app/model.joblib")
scaler = joblib.load("app/scaler.joblib")

THRESHOLD = 0.2

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]
    prediction = int(prob >= THRESHOLD)

    return {
        "fraud_probability": prob,
        "prediction": prediction
    }
