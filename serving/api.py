import mlflow.sklearn
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model.pkl")
@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    score = model.predict_proba(df)[0][1]
    return {
        "fraud_score": float(score),
        "is_fraud": bool(score > 0.5)
    }