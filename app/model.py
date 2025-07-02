import mlflow.pyfunc
import joblib
import os
import re

# ✅ Identifiants MLflow
RUN_ID = "55ad07bb940e476ebba44582b90ddc82"
EXPERIMENT_ID = "451208790809575449"

# ✅ Base path (relatif au fichier model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "mlruns", EXPERIMENT_ID, RUN_ID, "artifacts"))

# ✅ URI d'accès au modèle
MODEL_URI = os.path.join(MLRUNS_PATH, "model")
MLB_PATH = os.path.join(MLRUNS_PATH, "mlb.pkl")

# ⏳ Lazy load
pipeline = None
mlb = None

def load_model_once():
    global pipeline, mlb
    if pipeline is None or mlb is None:
        if not os.path.exists(MODEL_URI):
            raise FileNotFoundError(f"Modèle non trouvé à l'emplacement : {MODEL_URI}")
        if not os.path.exists(MLB_PATH):
            raise FileNotFoundError(f"Encodeur mlb.pkl non trouvé à l'emplacement : {MLB_PATH}")
        pipeline = mlflow.pyfunc.load_model(MODEL_URI)
        mlb = joblib.load(MLB_PATH)

# ✅ Nettoyage du texte
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# ✅ Prédiction avec chargement à la demande
def predict_tags(title: str):
    if not title.strip():
        return []
    load_model_once()
    cleaned_title = clean_text(title)
    prediction = pipeline.predict([cleaned_title])
    return mlb.inverse_transform(prediction)[0]
