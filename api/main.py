
# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd # Ajout de cette ligne

# 1. Initialiser l\"application FastAPI
app = FastAPI(title="Dubai Mall Customer Segmentation API")

# 2. Charger le modèle KMeans sauvegardé et le scaler
model_data = joblib.load("models/kmeans_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# 3. Définir le format des données envoyées par l\"utilisateur
class CustomerData(BaseModel):
    gender: str  # "Male" ou "Female"
    age: int
    annual_income_k: float
    spending_score: float

# 4. Créer un endpoint API
@app.post("/predict")
def predict_segment(customer: CustomerData):
    # Encoder le genre (comme à l\"entraînement)
    gender_encoded = 1 if customer.gender.lower() == "male" else 0

    # Construire le tableau de données pour la prédiction
    # Assurez-vous que l\"ordre des colonnes correspond à celui utilisé lors de l\"entraînement
    input_data = pd.DataFrame([{
        'Age': customer.age,
        'Annual Income (k$)': customer.annual_income_k,
        'Spending Score (1-100)': customer.spending_score
    }])

    # Appliquer le scaler sur les données numériques
    scaled_input_data = scaler.transform(input_data[features])

    # Prédire le cluster
    cluster = model.predict(scaled_input_data)[0]

    return {
        "cluster": int(cluster),
        "message": f"Le client appartient au segment {cluster}."
    }

# 5. Ajouter un endpoint de test
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l\"API de segmentation des clients du Dubai Mall 🚀"}


