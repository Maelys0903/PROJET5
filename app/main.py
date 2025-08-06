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
    try:
        if not question.text.strip():
            return {"tags": []}

        print("Texte reçu :", question.text)

        predicted_tags = bert.predict_tags(question.text)

        print("Tags prédits :", predicted_tags)

        return {"tags": predicted_tags}

    except Exception as e:
        print("Erreur pendant la prédiction :", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")