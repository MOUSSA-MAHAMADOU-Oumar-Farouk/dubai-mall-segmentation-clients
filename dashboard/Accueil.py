import streamlit as st

def run():
    st.title("🛍️ Segmentation Avancée des Clients du Dubai Mall")

    st.markdown("""
    <div style="border-left: 6px solid #2196F3; background-color: #E8F4FD; padding: 1rem; border-radius: 5px;">  
        <p style="text-align: justify;">  
    Bienvenue sur le tableau de bord dédié à l'analyse et à la segmentation de la clientèle du <b>Dubai Mall</b>. 
    Dans un secteur du commerce de détail en constante évolution, comprendre les comportements et les préférences des clients est essentiel. 
    Ce projet vise à transformer les données brutes en informations stratégiques pour optimiser les campagnes marketing et enrichir l'expérience client.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("--- ")

    st.header("Objectifs du projet")
    st.markdown("""
    Notre mission est de développer un modèle de segmentation client robuste et pertinent. 
    Ce modèle permettra aux gestionnaires du Dubai Mall de :
    *   **Mieux comprendre** la diversité de leur clientèle.
    *   **Adapter** leurs services et offres commerciales aux besoins spécifiques de chaque segment.
    *   **Optimiser** les stratégies de marketing pour une meilleure efficacité.
    *   **Améliorer** significativement la satisfaction et la fidélisation des clients.
    """)
    st.markdown("--- ")

    st.header("Étapes clés de notre approche")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div style="border-left: 6px solid #2196F3; background-color: #E8F4FD; height: 380px; padding: 1rem; border-radius: 5px;">    
        <h6>1. Analyse exploratoire des données (EDA)</h6>
        <p style="text-align: justify;">
        Plongée approfondie dans les données brutes pour découvrir des tendances, des anomalies et des relations, 
        formant la base de notre compréhension client.
        </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
         st.markdown("""
            <div style="border-left: 6px solid #2196F3; background-color: #E8F4FD; height: 380px; padding: 1rem; border-radius: 5px;">    
        <h6>2. Construction des modèles de clustering</h6>
            <p style="text-align: justify;">
        Développement des modèles de clustering pour regrouper les clients 
        en segments homogènes selon leurs caractéristiques.
            </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="border-left: 6px solid #2196F3; background-color: #E8F4FD; height: 380px; padding: 1rem; border-radius: 5px;">
                <h6>3. Construction de l'API de prédiction</h6>
                <p style="text-align: justify;">
                    Mise en place d'une interface de programmation permettant de prédire le segment d'un nouveau client, 
                    facilitant l'intégration dans les systèmes existants.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="border-left: 6px solid #2196F3; background-color: #E8F4FD; height: 380px; padding: 1rem; border-radius: 5px;">
                    
        <h6>4. Visualisation et interaction via Dashboard</h6>
        <p style="text-align: justify;">
            Création d'un tableau de bord interactif pour visualiser les segments de clientèle, 
            explorer les profils types et suivre les performances du modèle.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("--- ")

    st.markdown("""
    <div style="border-left: 6px solid #fcca46 ; background-color: #E8F4FD; padding: 1rem; border-radius: 5px;">  
        <p style="text-align: justify;"> 
    💡Ce projet est une initiative clé pour le Dubai Mall, visant à transformer l'expérience d'achat 
    et à renforcer la relation avec chaque client.
        <p/>
    </div>""", unsafe_allow_html=True)

    