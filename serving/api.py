from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Charge le modèle depuis le fichier local
with open("serving/model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running !"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    score = model.predict_proba(df)[0][1]
    is_fraud = bool(score > 0.5)
    return {
        "fraud_score": round(float(score), 4),
        "is_fraud": is_fraud,
        "risk_level": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.5 else "LOW"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "fraud_detector_v1"}