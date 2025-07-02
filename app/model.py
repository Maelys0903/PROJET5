import mlflow.pyfunc
import joblib
import os
import re

# ✅ Identifiants du run
RUN_ID = "55ad07bb940e476ebba44582b90ddc82"
EXPERIMENT_ID = "451208790809575449"

# ✅ URI d'accès au modèle (forme exigée par MLflow)
MODEL_URI = f"file:///C:/Mes documents/OpenClassRooms/PROJET5/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
MLB_PATH = f"C:/Mes documents/OpenClassRooms/PROJET5/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/mlb.pkl"

# ✅ Chargement du modèle et de l’encodeur
MLB_PATH = "mlruns/451208790809575449/55ad07bb940e476ebba44582b90ddc82/artifacts/mlb.pkl"
mlb = joblib.load(MLB_PATH)

# ✅ Nettoyage du texte
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# ✅ Prédiction à partir du texte nettoyé
def predict_tags(title: str):
    cleaned_title = clean_text(title)
    prediction = pipeline.predict([cleaned_title])
    return mlb.inverse_transform(prediction)[0]
