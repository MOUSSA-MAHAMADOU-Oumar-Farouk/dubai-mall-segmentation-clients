import streamlit as st
import subprocess
import time
import os
from dashboard import Accueil, EDA, Modelisation, Prediction

#st.set_page_config(page_title="Segmentation des clients du Dubai Mall")
#layout="wide", 
# Lancer l'API FastAPI automatiquement
#subprocess.Popen(["python", "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"])
subprocess.Popen([
    "python", "-m", "uvicorn", 
    "api.api:app", 
    "--reload",
    "--port", "8000"
])


# App Streamlit
PAGES = {
    "Accueil": Accueil,
    "Analyse exploratoire": EDA,
    "Modélisation": Modelisation,
    "Prédiction": Prediction
}

# Initialiser la session state pour la page courante
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Accueil"

# CSS pour styliser les boutons avec fond bleu
st.markdown("""
<style>
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
    margin-bottom: 10px;
    transition: background-color 0.3s;
}

.stButton > button:hover {
    background-color: #0d5aa7;
    color: white;
}

.stButton > button:focus {
    background-color: #0d5aa7;
    color: white;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")

# Créer les boutons de navigation
for page_name in PAGES.keys():
    if st.sidebar.button(page_name, key=f"btn_{page_name}"):
        st.session_state.current_page = page_name

# Exécuter la page courante
PAGES[st.session_state.current_page].run()

#selection = st.sidebar.radio("Aller à", list(PAGES.keys()))

#if selection != st.session_state.current_page:
#    st.session_state.current_page = selection

#PAGES[st.session_state.current_page].run()