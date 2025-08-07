from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app import bert

from app.bert import predict_tags

app = FastAPI()

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
        predicted_tags = predict_tags(question.text)
        return {"tags": predicted_tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
    