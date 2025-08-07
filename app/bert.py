# app/bert.py

import re
import joblib
import mlflow.pyfunc
import pandas as pd

from typing import List
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bert")
MLB_PATH = os.path.join(os.path.dirname(__file__), "mlb.pkl")

# Variables globales (non chargées au démarrage)
model = None
mlb = None

def load_model_once():
    """Charge le modèle BERT (MLflow) et le MultiLabelBinarizer une seule fois."""
    global model, mlb
    if model is None or mlb is None:
        model = mlflow.pyfunc.load_model(MODEL_PATH)
        mlb = joblib.load(MLB_PATH)

def clean_text(text: str) -> str:
    """Nettoie le texte d'entrée : minuscule + suppression ponctuation."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def predict_tags(text: str) -> list[str]:
    """Prédit les tags à partir d’un texte."""
    load_model_once()  # charge le modèle si nécessaire
    
    # Préparer l'entrée sous forme de DataFrame
    df = pd.DataFrame({"text": [text]})
    
    # Prédire les tags sous forme binaire
    preds = model.predict(df)  # np.ndarray shape (1, n_tags)

    # Convertir en liste des tags
    tags = mlb.inverse_transform(preds)
    return list(tags[0]) if tags else []
