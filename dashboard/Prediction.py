import streamlit as st
import pandas as pd
import requests
import json
from typing import Dict, Any

API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE_URL}/predict"

# Dictionnaire des personas
PERSONAS = {
    0: "Seniors stables",
    1: "Jeunes actifs équilibrés",
    2: "Adultes riches économes",
    3: "Jeunes aisés dépensiers",
    4: "Jeunes modestes impulsifs",
    5: "Adultes modestes économes"
}

def check_api_health() -> bool:
    """Vérifier si l'API est disponible"""
    try:
        response = requests.get(API_BASE_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def predict_single_customer(age: int, income: float, score: int) -> Dict[str, Any]:
    """Prédiction pour un client unique - Adapté à votre API"""
    data = {
        "Age": float(age),
        "Annual_Income": float(income),
        "Spending_Score": float(score)
    }
    
    try:
        response = requests.post(PREDICT_URL, json=data, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Erreur {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Erreur de connexion: {str(e)}"}

def predict_batch_customers(df: pd.DataFrame) -> Dict[str, Any]:
    """Prédiction pour plusieurs clients - Implémenté localement"""
    results = []
    for _, row in df.iterrows():
        data = {
            "Age": float(row["Age"]),
            "Annual_Income": float(row["Annual Income (k$)"]),
            "Spending_Score": float(row["Spending Score (1-100)"])
        }
        try:
            response = requests.post(PREDICT_URL, json=data, timeout=5)
            if response.status_code == 200:
                results.append(response.json())
            else:
                results.append({
                    "error": f"Erreur {response.status_code}",
                    "features": data
                })
        except Exception as e:
            results.append({
                "error": f"Exception: {str(e)}",
                "features": data
            })
    
    return {"success": True, "results": results}

def display_prediction_result(result: Dict[str, Any]):
    """Afficher les résultats de prédiction de manière formatée"""
    if result["success"]:
        data = result["data"]
        
        # Affichage du message de succès
        st.success("✅ Prédiction réussie !")
        
        # Récupération du cluster et du persona
        cluster_id = data.get("cluster", "Inconnu")
        persona = PERSONAS.get(cluster_id, "Persona inconnu")
        
        # Métriques principales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cluster ID", cluster_id)
        with col2:
            st.metric("Persona", persona)  
    else:
        st.error(f"Erreur: {result['error']}")

def run():
    
    st.title("Prédiction de segmentation client")
    st.markdown("---")
    
    # Vérification de l'état de l'API
    with st.spinner("Vérification de l'API..."):
        api_status = check_api_health()
    
    if not api_status:
        st.error("L'API n'est pas disponible. Veuillez démarrer le serveur FastAPI.")
        st.code("uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
        st.stop()
    
    st.success("API connectée avec succès")
    
    # Choix du type de prédiction
    choix = st.radio(
        "Choisir le type de prédiction", 
        ["Client unique", "Plusieurs clients (fichier à joindre)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if choix == "Client unique":
        st.subheader("Prédiction pour un client unique")
        
        # Formulaire de saisie
        with st.form("single_prediction"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input(
                    "Âge", 
                    min_value=18, 
                    max_value=100, 
                    value=35,
                    help="Âge du client entre 18 et 100 ans"
                )
            
            with col2:
                income = st.number_input(
                    "Revenu annuel (k$)", 
                    min_value=0.0, 
                    max_value=200.0,
                    value=65.0,
                    step=0.1,
                    help="Revenu annuel en milliers de dollars"
                )
            
            with col3:
                score = st.number_input(
                    "Score de dépenses (1-100)", 
                    min_value=1, 
                    max_value=100, 
                    value=75,
                    help="Score de dépenses entre 1 et 100"
                )
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            preview_data = {
                "Age": [age],
                "Annual Income (k$)": [income],
                "Spending Score": [score]
            }
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
            
            # Bouton de prédiction
            submitted = st.form_submit_button("Classer le client", type="primary")
            
            if submitted:
                with st.spinner("Prédiction en cours..."):
                    result = predict_single_customer(age, income, score)
                    display_prediction_result(result)
    
    else:
        st.subheader("Prédiction pour plusieurs clients")
        
        # Upload de fichier
        fichier = st.file_uploader(
            "📁 Uploader le fichier", 
            type=["xlsx", "csv"],
            help="Fichier Excel ou CSV avec colonnes: Age, Annual Income (k$), Spending Score (1-100)"
        )
        
        if fichier is not None:
            try:
                # Chargement du fichier
                if fichier.name.endswith('.xlsx'):
                    df = pd.read_excel(fichier)
                else:
                    df = pd.read_csv(fichier)
                
                st.success(f"Fichier chargé: {len(df)} lignes")
                
                # Vérification des colonnes requises
                required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Colonnes manquantes: {', '.join(missing_columns)}")
                    st.info("Colonnes requises: Age, Annual Income (k$), Spending Score (1-100)")
                else:
                    # Affichage de l'aperçu des données
                    st.subheader("Aperçu des données")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Statistiques descriptives
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nombre de clients", len(df))
                    with col2:
                        st.metric("Âge moyen", f"{df['Age'].mean():.1f}")
                    with col3:
                        st.metric("Revenu moyen", f"{df['Annual Income (k$)'].mean():.1f}k$")
                    with col4:
                        st.metric("Score moyen", f"{df['Spending Score (1-100)'].mean():.1f}")
                    
                    # Bouton de prédiction
                    if st.button("Lancer les prédictions", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Envoi des données à l'API...")
                        progress_bar.progress(30)
                        
                        # Appel de l'API
                        result = predict_batch_customers(df)
                        progress_bar.progress(100)
                        
                        if result["success"]:
                            st.success("Prédictions terminées avec succès !")
                            api_results = result["results"]
                            
                            # Ajout des résultats au dataframe
                            df_results = df.copy()
                            clusters = []
                            personas = []
                            
                            for r in api_results:
                                cluster_id = r.get("cluster", -1)
                                clusters.append(cluster_id)
                                personas.append(PERSONAS.get(cluster_id, "Inconnu"))
                            
                            df_results["Cluster"] = clusters
                            df_results["Persona"] = personas
                            
                            # Résultats
                            st.subheader("Résultats des prédictions")
                            st.dataframe(df_results, use_container_width=True)
                            
                            # Statistiques des clusters
                            st.subheader("Répartition par cluster")
                            cluster_counts = df_results["Persona"].value_counts()
                            st.bar_chart(cluster_counts)
                            
                            # Option de téléchargement
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="💾 Télécharger les résultats",
                                data=csv,
                                file_name="predictions_clients.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.error(f"Erreur lors des prédictions: {result.get('error', 'Erreur inconnue')}")
                        
                        status_text.empty()
                        progress_bar.empty()
                        
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

if __name__ == "__main__":
    run()