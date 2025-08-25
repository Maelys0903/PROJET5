# Projet 5 - API de prédiction de mots clés

## Description

Ce projet consiste à développer une API REST utilisant FastAPI, capable de prédire des mots clés (tags) à partir d’un texte donné. L’objectif est d’automatiser la suggestion de tags pertinents pour faciliter le classement et la recherche de contenus.

Le projet intègre également une chaîne d’intégration continue (CI) avec GitHub Actions pour exécuter automatiquement les tests unitaires à chaque push vers le dépôt GitHub.

## Structure du projet

PROJET5/
├── .github/                    # Configuration CI/CD GitHub Actions
│   └── workflows/
│       └── python-app.yml      # Pipeline de déploiement continu + tests unitaires
│
├── app/                        # Code source de l'API
│   ├── __pychase__/            # Fichiers compilés Python (générés automatiquement)
│   ├── model_bert/             # Dossier contenant le modèle BERT sauvegardé
│   ├── __init__.py             # Rend le dossier utilisable comme package Python
│   ├── bert.py                 # Implémentation du modèle BERT + fonctions de prédiction
│   ├── config.py               # Configuration 
│   ├── main.py                 # Point d'entrée de l'API (FastAPI / Flask)    
│   ├── mlb.pkl                 # Encodage MultiLabelBinarizer (pour les tags de sortie)
│   ├── runtime.txt             # Version de Python spécifiée pour le déploiement (Render)
│
├── mlruns/
│
├── tests/                      # Tests unitaires
│   ├── __pychase__/            # Fichiers compilés Python (tests)
│   ├── test_predict.py         # Tests sur les fonctions de prédiction
│
├── venv310/                    # Environnement virtuel local
│
├── .gitignore
│
├── Dubreuil_Maëlys_1_notebook_exploration_082025.ipynb
├── Dubreuil_Maëlys_2_notebook_requête_API_082025.ipynb 
├── Dubreuil_Maëlys_3_notebook_non_supervisée_082025.ipynb
├── Dubreuil_Maëlys_4_notebook_supervisée1_082025.ipynb   
├── Dubreuil_Maëlys_4_notebook_supervisée2_082025.ipynb 
├── Dubreuil_Maëlys_4_notebook_supervisée3_082025.ipynb 
│
├── mlb.plk
├── QueryResults.csv            # Jeu de données
├── README.md                   # Objectif du projet + explication structure
├── render.yaml                 # Configuration de déploiement sur Render
├── requirements.txt            # Liste des packages
├── save_model_bert.py          # Script pour sauvegarder le modèle BERT entraîné
├── stack_questions_api.csv     # Jeu de données

## Activer l'environnement via PowerShell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

& "C:/Mes documents/OpenClassRooms_Machine_Learning_Engineer/Catégorisez_automatiquement_des_questions_Dubreuil_Maëlys/PROJET5/venv310/Scripts/Activate.ps1"

## Lancer l'API localement
uvicorn app.main:app --reload

{
  "text": "Difference between Python list and tuple"
}

## Lancer vers GitHub Actions et Render
git add .
git commit -m "Mise à jour du modèle BERT et API"
git push