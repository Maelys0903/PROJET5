from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.model import predict_tags

app = FastAPI(
    title="API de Prédiction de Mots-clés",
    description="Prévoit les mots-clés associés à un texte donné",
    version="1.0"
)

# Classe d'entrée pour validation des données
class TextInput(BaseModel):
    text: str

# Classe de sortie pour formatage de la réponse
class PredictionResponse(BaseModel):
    keywords: List[str]

# Route principale de prédiction
@app.post("/predict", response_model=PredictionResponse)
def predict_keywords(input: TextInput):
    predicted = predict_tags(input.text) # Appelle le modèle ML
    return {"keywords": predicted}
