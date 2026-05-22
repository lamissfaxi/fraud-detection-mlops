# 🔍 Fraud Detection MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.12-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> Pipeline MLOps complet de détection de fraude bancaire en temps réel — de l'ingestion des données jusqu'au déploiement en container Docker.

---

## 📊 Résultats

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | **0.97** |
| F1-Score (classe fraude) | **0.85+** |
| Latence API | **< 50ms** |
| Dataset | 284 807 transactions |
| Taux de fraude réel | 0.17% (classes déséquilibrées) |

---

## 🏗️ Architecture

```
Transactions bancaires
        │
        ▼
┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data & EDA   │────▶│  ML Training │────▶│   MLflow    │
│  (Pandas)     │     │  (XGBoost)   │     │  Registry   │
└───────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │   FastAPI       │
                                        │  /predict       │
                                        │  < 50ms         │
                                        └─────────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │  Docker         │
                                        │  Container      │
                                        └─────────────────┘
```

---

## 🚀 Lancer le projet en 2 minutes

### Avec Docker (recommandé)

```bash
# 1. Clone le projet
git clone https://github.com/lamissfaxi/fraud-detection-mlops.git
cd fraud-detection-mlops

# 2. Lance le container
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api

# 3. Ouvre l'API
# http://localhost:8000/docs
```

### Sans Docker (local)

```bash
# 1. Clone et installe
git clone https://github.com/lamissfaxi/fraud-detection-mlops.git
cd fraud-detection-mlops
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 2. Lance l'API
uvicorn serving.api:app --reload

# 3. Ouvre l'API
# http://localhost:8000/docs
```

---

## 📁 Structure du projet

```
fraud-detection-mlops/
│
├── data/                        # Scripts d'analyse (CSV non inclus)
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_time_distribution.png
│   └── 04_features_distribution.png
│
├── training/                    # Pipeline ML
│   ├── train.py                 # Entraînement XGBoost + MLflow tracking
│   └── export_model.py          # Export modèle en .pkl
│
├── serving/                     # API de prédiction
│   ├── api.py                   # FastAPI endpoint /predict
│   └── model/
│       └── fraud_model.pkl      # Modèle entraîné
│
├── mlruns/                      # Expériences MLflow (local)
│
├── Dockerfile                   # Containerisation
├── requirements.txt             # Dépendances Python
└── README.md
```

---

## 🤖 Pipeline Machine Learning

### 1. Données
- **Dataset** : [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle
- **284 807 transactions** sur 2 jours
- **492 fraudes** seulement (0.17%) → problème de classes déséquilibrées

### 2. Preprocessing
- Analyse exploratoire complète (EDA)
- **SMOTE** pour rééquilibrer les classes
- Normalisation des features `Amount` et `Time`
- Features V1–V28 issues d'une PCA (données anonymisées)

### 3. Modèle
- **XGBoost** avec tuning des hyperparamètres
- Tracking de toutes les expériences avec **MLflow**
- Métriques : AUC-ROC, F1-Score, Precision-Recall

### 4. Serving
- **FastAPI** avec endpoint `/predict`
- Réponse JSON avec score de fraude + niveau de risque
- Documentation automatique Swagger UI

---

## 🌐 API — Endpoints

### `GET /health`
Vérifie que l'API tourne.

```json
{"status": "ok", "model": "fraud_detector_v1"}
```

### `POST /predict`
Prédit si une transaction est frauduleuse.

**Exemple de requête :**
```json
{
  "Time": 406,
  "V1": -2.3122265423,
  "V2": 1.9519999746,
  "V3": -1.6098003838,
  "Amount": 149.62,
  "...": "..."
}
```

**Réponse :**
```json
{
  "fraud_score": 0.9823,
  "is_fraud": true,
  "risk_level": "HIGH"
}
```

---

## 🛠️ Stack technique

| Catégorie | Technologie |
|-----------|-------------|
| Langage | Python 3.11 |
| Machine Learning | XGBoost, Scikit-learn |
| MLOps / Tracking | MLflow |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn |
| Containerisation | Docker |
| Versioning | Git / GitHub |

---

## 📈 Résultats EDA

Les analyses exploratoires ont révélé :
- Les fraudes représentent seulement **0.17%** des transactions
- Le montant moyen d'une fraude est **différent** des transactions normales
- Les features **V14, V12, V10, V11, V4** sont les plus discriminantes
- Aucune valeur manquante dans le dataset

---

## 👩‍💻 Auteure

**Lamis Sfaxi**
Étudiante ingénieure — Big Data & Intelligence Artificielle

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lamis_Sfaxi-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/lamis-sfaxi)
[![GitHub](https://img.shields.io/badge/GitHub-lamissfaxi-181717?style=flat-square&logo=github)](https://github.com/lamissfaxi)

---

## 📄 Licence

Ce projet est sous licence MIT — libre d'utilisation pour des fins éducatives.
