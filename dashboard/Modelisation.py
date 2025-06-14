import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run():
    df = pd.read_csv("data/Mall_Customers.csv")
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering KMeans
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Personas
    personas_dict = {
        0: "Seniors stables",
        1: "Jeunes actifs équilibrés",
        2: "Adultes riches économes",
        3: "Jeunes aisés dépensiers",
        4: "Jeunes modestes impulsifs",
        5: "Adultes modestes économes"
    }
    df["ClusterLabel"] = df["Cluster"].map(personas_dict)

    st.title("Modélisation - Clustering KMeans")

    st.info("""
    **Modèle retenu : KMeans**\n
    Après évaluation comparative des algorithmes DBSCAN, Agglomératif et KMeans, ce dernier a été retenu pour ses bonnes performances et la clarté des segments générés. Il permet une interprétation utile en marketing.
    """)

    # Scores
    st.subheader("📊 Indicateurs de performance du modèle")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div style='background-color:#e3f2fd; padding:10px; border-radius:5px; text-align:center;'>"
                    "<strong>Clusters</strong><br><h3>6</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color:#e8f5e9; padding:10px; border-radius:5px; text-align:center;'>"
                    "<strong>Silhouette</strong><br><h3>0.428</h3></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='background-color:#fff3e0; padding:10px; border-radius:5px; text-align:center;'>"
                    "<strong>Davies-Bouldin</strong><br><h3>0.825</h3></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div style='background-color:#ede7f6; padding:10px; border-radius:5px; text-align:center;'>"
                    "<strong>Calinski-Harabasz</strong><br><h3>135.1</h3></div>", unsafe_allow_html=True)

    # Visualisations interactives
    st.subheader("🌀 Visualisation des clusters")

    if "view" not in st.session_state:
        st.session_state.view = ""

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("PCA"):
            st.session_state.view = "PCA" if st.session_state.view != "PCA" else ""
    with col2:
        if st.button("t-SNE"):
            st.session_state.view = "TSNE" if st.session_state.view != "TSNE" else ""
    with col3:
        if st.button("PCA + t-SNE"):
            st.session_state.view = "BOTH" if st.session_state.view != "BOTH" else ""

    if st.session_state.view in ["PCA", "BOTH"]:
        st.markdown("#### Visualisation PCA")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
        ax.set_title("KMeans - Vue PCA")
        st.pyplot(fig)

    if st.session_state.view in ["TSNE", "BOTH"]:
        st.markdown("#### Visualisation t-SNE")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X_scaled)
        df["TSNE1"], df["TSNE2"] = tsne_result[:, 0], tsne_result[:, 1]
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="Cluster", palette="Set2", ax=ax)
        ax.set_title("KMeans - Vue t-SNE")
        st.pyplot(fig)

    # Analyse des personas
    st.subheader("👥 Analyse des personas")
    personas = df.groupby(["Cluster", "ClusterLabel"])[features].mean().round(1).reset_index()
    personas["Effectif"] = df.groupby("Cluster").size().values
    st.dataframe(personas)

    # Répartition par genre
    st.subheader("🚻 Répartition Hommes/Femmes par cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    gender_counts = df.groupby(["ClusterLabel", "Gender"]).size().reset_index(name='count')
    gender_counts["percent"] = gender_counts.groupby("ClusterLabel")["count"].transform(lambda x: x / x.sum())
    sns.barplot(
        data=gender_counts,
        y="ClusterLabel",
        x="percent",
        hue="Gender",
        palette="Set2",
        ax=ax,
        orient="h"
    )
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.0%}" for v in container.datavalues], label_type='center', fontsize=10, color='black')
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Cluster")
    ax.set_title("Répartition H/F par cluster (en %)")
    ax.legend(title="Genre")
    st.pyplot(fig)

    st.subheader("Visualisation 3D")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = sns.color_palette("Set2", 6)

    for i in range(6):
        cluster_data = df[df["Cluster"] == i]
        ax.scatter(
            cluster_data["Age"],
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            label=personas_dict[i],
            color=colors[i],
            s=50
        )

    ax.set_xlabel("Âge")
    ax.set_ylabel("Revenu annuel (k$)")
    ax.set_zlabel("Score de dépense")
    ax.set_title("Clusters - Vue 3D")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""   
        <div style="border-left: 6px solid #2196F3 ; margin : 10px; background-color: #E8F4FD; padding: 1px; border-radius: 5px;">  
        <h3>Recommandations stratégiques</h3>
                
    </div>""", unsafe_allow_html=True)
                
    st.markdown("""
    <div style="border-left: 6px solid #fcca46 ; width :705px; background-color: #E8F4FD; padding: 1rem; border-radius: 5px;">  
        <p style="text-align: justify;"> 
    <strong>1️⃣ Cibler le segment "Jeunes aisés dépensiers"</strong> avec des offres haut de gamme et des programmes de fidélité premium.
        <p/>
        <p style="text-align: justify;">         
    <strong>2️⃣ Proposer des offres d'épargne ou de réduction aux adultes riches économes</strong> pour stimuler leur engagement.
        <p/> 
        <p style="text-align: justify;"> 
    <strong>3️⃣ Automatiser des campagnes marketing spécifiques</strong> par cluster via des canaux adaptés (SMS,   email, réseaux sociaux).
    <p/>
    <p style="text-align: justify;"> 
    <strong>4️⃣ Personnaliser l'expérience client</strong> en boutique ou en ligne selon le cluster (recommandations produits, messages, visuels).
    <p/> 
    <p style="text-align: justify;">           
    <strong>5️⃣ Surveiller les évolutions comportementales</strong> pour mettre à jour les clusters périodiquement.
     <p/>
    </div>""", unsafe_allow_html=True)
