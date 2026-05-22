import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/creditcard.csv"
RANDOM_STATE = 42

mlflow.set_experiment("fraud-detection")


# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# scale
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = sm.fit_resample(X_train, y_train)


# -----------------------
# TRAIN + MLflow
# -----------------------
with mlflow.start_run(run_name="xgboost_baseline"):

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)

    # logs
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("f1", f1)

    # log model + REGISTER (IMPORTANT)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="fraud_model"
    )

    print(f"AUC={auc:.4f} | F1={f1:.4f}")