# Projet 5 - API de prédiction de mots clés

## Description

Ce projet consiste à développer une API REST utilisant FastAPI, capable de prédire des mots clés (tags) à partir d’un texte donné. L’objectif est d’automatiser la suggestion de tags pertinents pour faciliter le classement et la recherche de contenus.

Le projet intègre également une chaîne d’intégration continue (CI) avec GitHub Actions pour exécuter automatiquement les tests unitaires à chaque push vers le dépôt GitHub.

---

## Structure du projet

```plaintext
projet5-api/
│
├── app/
│   ├── main.py          # Point d'entrée de l'API FastAPI
│   ├── model.py         # Implémentation de la fonction de prédiction
│   ├── utils.py         # Fonctions utilitaires
│   └── __init__.py
│
├── tests/
│   └── test_main.py     # Tests unitaires pour l'API
│
├── requirements.txt     # Liste des packages Python requis
├── README.md            # Ce fichier
├── .github/
│   └── workflows/
│       └── pyhton-app.yml       # Pipeline GitHub Actions pour la 
