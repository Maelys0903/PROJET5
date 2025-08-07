# app/bert.py

import re
import joblib
import mlflow.pyfunc
import pandas as pd

from typing import List

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bert")
MLB_PATH = os.path.join(os.path.dirname(__file__), "mlb.pkl")

# Variables globales
pipeline = None
mlb = None

def load_model_once():
    """Charge le modèle BERT (depuis le Model Registry) et le MultiLabelBinarizer une seule fois."""
    global pipeline, mlb
    if pipeline is None or mlb is None:
        # Chargement depuis le Model Registry
        pipeline = mlflow.pyfunc.load_model(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)

def clean_text(text: str) -> str:
    """Nettoie le texte d'entrée : minuscule + suppression ponctuation."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# Chargement du modèle MLflow
model = mlflow.pyfunc.load_model("app/model_bert")

# Chargement du MultiLabelBinarizer pour décoder les prédictions
mlb = joblib.load("app/mlb.pkl")

def predict_tags(text: str) -> list[str]:
    # Préparer l'entrée sous forme DataFrame
    df = pd.DataFrame({"text": [text]})
    
    # Prédire le binaire multi-label (matrice 1xN)
    preds = model.predict(df)  # np.ndarray shape (1, n_tags)
    
    # Convertir en liste des tags prédits
    tags = mlb.inverse_transform(preds)
    # inverse_transform renvoie une liste de tuples, on prend le premier élément
    return list(tags[0]) if tags else []