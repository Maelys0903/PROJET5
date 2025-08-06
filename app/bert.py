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

def predict_tags(text: str) -> List[str]:
    try:
        if not text.strip():
            return []
        load_model_once()
        cleaned = clean_text(text)
        df_input = pd.DataFrame({"text": [cleaned]})
        prediction = pipeline.predict(df_input)
        return mlb.inverse_transform(prediction)[0]
    except Exception as e:
        print(f"Erreur dans predict_tags : {e}")
        raise