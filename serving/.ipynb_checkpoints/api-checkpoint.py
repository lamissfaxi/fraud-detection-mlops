from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="Fraud Detection API")

# Charger modèle MLflow (version locale pour éviter erreurs)
model = mlflow.pyfunc.load_model("models:/fraud_model/Production")

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    score = model.predict(df)[0]

    return {
        "fraud_score": float(score),
        "is_fraud": bool(score > 0.5)
    }