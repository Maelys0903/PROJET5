import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

import numpy as np

def test_predict_valid(monkeypatch):
    class MockPipeline:
        def predict(self, input_df):
            # Crée un vecteur binaire de taille 50 (avec 3 premiers à 1, le reste à 0)
            arr = np.zeros(50, dtype=int)
            arr[:3] = 1
            return arr.reshape(1, -1)

    def mock_load_model(model_uri):
        return MockPipeline()

    monkeypatch.setattr("mlflow.pyfunc.load_model", mock_load_model)

    payload = {"text": "How to create an API with FastAPI?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Vérifie que la réponse contient bien une liste de tags
    assert "tags" in data
    assert isinstance(data["tags"], list)
    assert len(data["tags"]) > 0


def test_predict_invalid_json():
    # Envoyer un JSON mal formé (champ text manquant)
    payload = {"wrong_field": "No text here"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # erreur de validation Pydantic


def test_predict_empty_text(monkeypatch):

    class MockPipeline:
        def predict(self, input_df):
            arr = np.zeros((1, 50), dtype=int)
            arr[0, :3] = 1  # Simuler 3 tags actifs
            return arr


    def mock_load_model(model_uri):
        return MockPipeline()

    monkeypatch.setattr("mlflow.pyfunc.load_model", mock_load_model)

    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "tags" in json_data
    assert json_data["tags"] == []


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction de tags !"}


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text