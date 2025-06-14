from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Charger le modèle et le scaler
model_data = joblib.load("models/kmeans_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# Dictionnaire des personas
personas = {
    0: "Seniors stables",
    1: "Jeunes actifs équilibrés",
    2: "Adultes riches économes",
    3: "Jeunes aisés dépensiers",
    4: "Jeunes modestes impulsifs",
    5: "Adultes modestes économes"
}

app = FastAPI(title="Customer Clustering API")

class CustomerData(BaseModel):
    Age: float
    Annual_Income: float
    Spending_Score: float

@app.post("/predict", summary="Prédire le cluster d'un client")
async def predict_cluster(customer: CustomerData):
    # Créer un DataFrame avec les données d'entrée
    input_data = pd.DataFrame([[
        customer.Age,
        customer.Annual_Income,
        customer.Spending_Score
    ]], columns=features)
    
    # Appliquer le scaling
    scaled_data = scaler.transform(input_data)
    
    # Faire la prédiction
    cluster = model.predict(scaled_data)[0]
    
    # Récupérer le nom du persona
    persona = personas.get(cluster, "Inconnu")
    
    return {
        "cluster": int(cluster),
        "persona": persona,
        "features": {
            "Age": customer.Age,
            "Annual_Income": customer.Annual_Income,
            "Spending_Score": customer.Spending_Score
        }
    }

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de clusters clients"}