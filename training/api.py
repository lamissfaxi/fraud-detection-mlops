from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Fraud Detection API")
model = mlflow.sklearn.load_model("models:/fraud_model/Production")

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    score = model.predict_proba(df)[0][1]
    return {"fraud_score": round(float(score), 4),
            "is_fraud": score > 0.5}