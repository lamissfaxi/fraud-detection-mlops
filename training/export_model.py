import mlflow
import pickle
import os

# Charge le modèle depuis MLflow local
model = mlflow.sklearn.load_model("models:/fraud_model/2")

# Sauvegarde en .pkl
os.makedirs("serving/model", exist_ok=True)
with open("serving/model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle exporté : serving/model/fraud_model.pkl")