from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from app.bert import predict_tags

app = FastAPI(
    title="API de Prédiction de Tags",
    description="API basée sur un modèle BERT + LogisticRegression pour prédire les tags d'une question StackOverflow.",
    version="1.0"
)

class Question(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    tags: List[str]

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de tags !"}

@app.post("/predict", response_model=PredictionResponse)
def predict(question: Question):
    if not question.text.strip():
        return {"tags": []}
    predicted_tags = predict_tags(question.text)
    return {"tags": predicted_tags}