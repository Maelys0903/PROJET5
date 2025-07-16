# app/bert.py

import re
import joblib
import mlflow.pyfunc

from typing import List

# Nom du modèle dans le MLflow Model Registry
MODEL_NAME = "model"
MODEL_STAGE = "Production"
MLB_PATH = "mlb.pkl"

# Variables globales
pipeline = None
mlb = None

def load_model_once():
    """Charge le modèle BERT (depuis le Model Registry) et le MultiLabelBinarizer une seule fois."""
    global pipeline, mlb
    if pipeline is None or mlb is None:
        # Chargement depuis le Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        pipeline = mlflow.pyfunc.load_model(model_uri)
        mlb = joblib.load(MLB_PATH)

def clean_text(text: str) -> str:
    """Nettoie le texte d'entrée : minuscule + suppression ponctuation."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def predict_tags(text: str) -> List[str]:
    """Prédit les tags à partir du texte d'entrée."""
    if not text.strip():
        return []
    load_model_once()
    cleaned = clean_text(text)
    df_input = {"text": [cleaned]} 
    prediction = pipeline.predict(df_input)
    return mlb.inverse_transform(prediction)[0]