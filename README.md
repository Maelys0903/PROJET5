# Projet 5 - API de prédiction de mots clés

## Description

Ce projet consiste à développer une API REST utilisant FastAPI, capable de prédire des mots clés (tags) à partir d’un texte donné. L’objectif est d’automatiser la suggestion de tags pertinents pour faciliter le classement et la recherche de contenus.

Le projet intègre également une chaîne d’intégration continue (CI) avec GitHub Actions pour exécuter automatiquement les tests unitaires à chaque push vers le dépôt GitHub.

## Structure du projet

project_root/
│
├── app/                        # Code source de l'API
│   ├── main.py                 # Point d'entrée de l'API (FastAPI / Flask)
│   ├── bert.py                 # Implémentation du modèle BERT + fonctions de prédiction
│   ├── config.py               # Configuration
│   ├── __init__.py             
│
├── tests/                      # Tests unitaires
│   └── test_predict.py         # Tests sur les fonctions de prédiction
│
├── requirements.txt            # Liste des packages
├── README.md                   # Objectif du projet + explication structure
├── .gitignore
├── .github/                    # Configuration CI/CD GitHub Actions
│   └── workflows/
│       └── python-app.yml      # Pipeline de déploiement continu + tests unitaires
|
├── Exploration.ipynb
├── API.ipynb 
├── Non_supervisée.ipynb
├── Supervisée.ipynb   
├── Supervisée2_BERT.ipynb 
├── Supervisée3_USE.ipynb 

## Lancer l'API localement
uvicorn app.main:app --reload

{
  "text": "Comment créer une API avec FastAPI et Python ?"
}

## Lancer l'API sur le cloud
Git hub & Render :
Via le terminal (exemple):
>> git add requirements.txt                                           
>> git commit -m "Mise en place du workflow GitHub Actions"
>> git push --set-upstream origin master

Render :
https://projet5-api-5qd0.onrender.com
Dans PowerShell :
>>   Invoke-RestMethod `
>>   -Uri "https://projet5-api-5qd0.onrender.com/predict" `
>>   -Method POST `
>>   -Headers @{"Content-Type" = "application/json"} `
>>   -Body '{ "text": "Comment installer FastAPI ?" }'
>>
Résultat :
tags
----
{python, fastapi}