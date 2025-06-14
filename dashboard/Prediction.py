
import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

def run():
    st.title("Prédiction de cluster")

    # Charger le scaler pour la prédiction
    try:
        model_data = joblib.load("models/kmeans_model.pkl")
        scaler = model_data["scaler"]
        features = model_data["features"]
    except FileNotFoundError:
        st.error("Modèle non trouvé. Veuillez entraîner le modèle d'abord.")
        return

    choix = st.radio("Choisir le type de prédiction", ["Client unique", "Plusieurs clients (Excel)"])

    if choix == "Client unique":
        gender = st.selectbox("Genre", ["Male", "Female"])
        age = st.number_input("Âge", 18, 100)
        income = st.number_input("Revenu annuel (k$)", 0.0)
        score = st.number_input("Score de dépenses", 0.0, 100.0)

        if st.button("Prédire"):
            # Préparer les données pour le scaler
            input_df = pd.DataFrame([{
                'Age': age,
                'Annual Income (k$)': income,
                'Spending Score (1-100)': score
            }])
            
            # Appliquer le scaler
            scaled_input = scaler.transform(input_df[features])

            # Encoder le genre (comme à l'entraînement)
            gender_encoded = 1 if gender.lower() == "male" else 0

            # Construire le tableau de données pour l'API (l'API gérera le scaling)
            data_for_api = {
                "gender": gender,
                "age": age,
                "annual_income_k": income,
                "spending_score": score
            }
            
            res = requests.post(API_URL, json=data_for_api)
            if res.status_code == 200:
                st.success(f"Prédiction : {res.json()['message']}")
            else:
                st.error(f"Erreur lors de la prédiction : {res.status_code} - {res.text}")

    else:
        fichier = st.file_uploader("Uploader le fichier Excel", type=["xlsx", "csv"])
        if fichier is not None:
            df = pd.read_excel(fichier) if fichier.name.endswith("xlsx") else pd.read_csv(fichier)
            st.dataframe(df)

            results = []
            for index, row in df.iterrows():
                data_for_api = {
                    "gender": row["Gender"],
                    "age": int(row["Age"]),
                    "annual_income_k": float(row["Annual Income (k$)"]),
                    "spending_score": float(row["Spending Score (1-100)"])
                }
                res = requests.post(API_URL, json=data_for_api)
                if res.status_code == 200:
                    cluster = res.json()["cluster"]
                    results.append(cluster)
                else:
                    results.append(f"Erreur: {res.status_code}")

            df["Cluster"] = results
            st.success("Prédiction terminée !")
            st.dataframe(df)


