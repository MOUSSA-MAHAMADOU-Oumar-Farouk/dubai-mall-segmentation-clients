# Projet de Segmentation Client du Dubai Mall

## Vue d'ensemble du projet

Ce projet vise à développer une solution complète de segmentation client pour le Dubai Mall, en utilisant des techniques de machine learning (clustering) et en fournissant une interface interactive via un tableau de bord Streamlit et une API FastAPI. L'objectif principal est de transformer les données brutes des clients en informations stratégiques, permettant au Dubai Mall de mieux comprendre sa clientèle, d'adapter ses offres et d'optimiser ses stratégies marketing pour une meilleure fidélisation.

## Fonctionnalités

Le projet offre les fonctionnalités clés suivantes :

*   **Analyse Exploratoire des Données (EDA)** : Visualisation et compréhension des caractéristiques des clients (âge, revenu annuel, score de dépenses, genre).
*   **Modélisation de Clustering** : Application de l'algorithme K-Means pour regrouper les clients en segments homogènes (personas).
*   **API de Prédiction** : Une API RESTful (FastAPI) permettant de prédire le segment d'un nouveau client en temps réel.
*   **Tableau de Bord Interactif** : Une application web (Streamlit) pour visualiser les résultats de la segmentation, explorer les personas et effectuer des prédictions pour des clients individuels ou en lot.

## Installation

Pour installer et exécuter ce projet localement, suivez les étapes ci-dessous :

### Prérequis

Assurez-vous d'avoir Python 3.8+ installé sur votre système.

### Cloner le dépôt

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
```

### Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows, utilisez `.\.venv\Scripts\activate`
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Démarrer l'API FastAPI

Ouvrez un terminal et exécutez la commande suivante à la racine du projet :

```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible à l'adresse `http://127.0.0.1:8000`. Vous pouvez consulter la documentation interactive de l'API à `http://127.0.0.1:8000/docs`.

### Démarrer le tableau de bord Streamlit

Ouvrez un **nouveau** terminal (en gardant l'API en cours d'exécution) et exécutez la commande suivante à la racine du projet :

```bash
streamlit run app.py
```

Le tableau de bord s'ouvrira automatiquement dans votre navigateur web. Si ce n'est pas le cas, accédez à `http://localhost:8501`.

## Structure du projet

```
.├── .venv/                   # Environnement virtuel
├── api/                     # Contient l'API FastAPI
│   ├── __pycache__/
│   └── api.py               # Point d'entrée de l'API
├── dashboard/               # Contient les différentes pages du tableau de bord Streamlit
│   ├── __pycache__/
│   ├── Accueil.py           # Page d'accueil
│   ├── EDA.py               # Page d'analyse exploratoire des données
│   ├── Modelisation.py      # Page de modélisation et visualisation des clusters
│   └── Prediction.py        # Page de prédiction client
├── data/                    # Contient les données brutes
│   └── Mall_Customers.csv   # Jeu de données des clients
├── models/                  # Contient le modèle retenu
│   └── kmeans_model.pkl     # Modèle K-Means sérialisé
├── scripts/                 # Scripts de développement et d'entraînement
│   ├── projet_ml_2025.py    # Script principal de développement ML (EDA, modélisation, évaluation)
│   ├── Notebook_ML_2025.ipynb #Notebook du projet
├── app.py                   # Point d'entrée principal de l'application Streamlit
├── requirements.txt         # Liste des dépendances Python
├── Segmentation_client_du_Dubai_Mall.pptx # Présentation du projet
└── README.md                # Ce fichier
```

## Technologies utilisées

*   **Python** : Langage de programmation principal.
*   **Pandas & NumPy** : Manipulation et analyse de données.
*   **Scikit-learn** : Algorithmes de Machine Learning (K-Means, PCA, t-SNE).
*   **Matplotlib & Seaborn & Plotly** : Visualisation de données.
*   **Streamlit** : Création du tableau de bord interactif.
*   **FastAPI** : Développement de l'API RESTful.
*   **Uvicorn** : Serveur ASGI pour FastAPI.
*   **Joblib** : Sérialisation et désérialisation du modèle K-Means.

## Auteurs

*   **MOUSSA MAHAMADOU OUMAR FAROUK**
*   **W. ABIJA DEBORAH MANDO**
*   **ZAONGO INOUSSA**
*   **ZOUMANA ZERBO**


