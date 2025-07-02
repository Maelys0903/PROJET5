import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from app.main import app

# Utilisé pour remplacer temporairement predict_tags
from unittest.mock import patch

client = TestClient(app)

# Simulation de prédiction réussie
@patch("app.main.predict_tags", return_value=["python", "data-science"])
def test_predict_valid_input(mock_predict):
    response = client.post("/predict", json={"text": "How to learn Python for data science?"})
    assert response.status_code == 200
    assert response.json() == {"keywords": ["python", "data-science"]}

# Simulation si le champ est vide
@patch("app.main.predict_tags", return_value=[])
def test_predict_empty_input(mock_predict):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    assert response.json() == {"keywords": []}
