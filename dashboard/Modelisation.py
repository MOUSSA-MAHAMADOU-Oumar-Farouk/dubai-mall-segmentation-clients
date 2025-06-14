
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import product
import scipy.cluster.hierarchy as sch

from src.data_preprocessing import load_and_preprocess_data

def run():
    st.title("Modélisation et Clustering")

    file_path = "data/Mall_Customers.csv"
    df, scaler = load_and_preprocess_data(file_path)
    cols_for_clustering = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    df_scaled = df[cols_for_clustering]

    st.subheader("Clustering Hiérarchique (Agglomératif)")

    st.markdown("### Dendrogramme")
    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
    dendrogram = linkage(df_scaled, method='ward')
    sch.dendrogram(dendrogram, ax=ax_dendro)
    ax_dendro.set_title("Dendrogramme (Méthode de Ward)")
    ax_dendro.set_xlabel("Indices des Clients")
    ax_dendro.set_ylabel("Distance Euclidienne")
    st.pyplot(fig_dendro)

    st.markdown("### Visualisation des clusters à travers PCA")
    clustering_agg = AgglomerativeClustering(n_clusters=6)
    cluster_labels_agg = clustering_agg.fit_predict(df_scaled)
    pca_agg = PCA(n_components=2)
    X_pca_agg = pca_agg.fit_transform(df_scaled)
    fig_pca_agg, ax_pca_agg = plt.subplots(figsize=(8, 5))
    ax_pca_agg.scatter(X_pca_agg[:, 0], X_pca_agg[:, 1], c=cluster_labels_agg, cmap='viridis')
    ax_pca_agg.set_title("Clusters hiérarchiques projetés via PCA")
    ax_pca_agg.set_xlabel("Composante principale 1")
    ax_pca_agg.set_ylabel("Composante principale 2")
    st.pyplot(fig_pca_agg)

    st.markdown("### Visualisation des clusters à travers t-SNE")
    tsne_agg = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne_agg = tsne_agg.fit_transform(df_scaled)
    fig_tsne_agg, ax_tsne_agg = plt.subplots(figsize=(8, 5))
    ax_tsne_agg.scatter(X_tsne_agg[:, 0], X_tsne_agg[:, 1], c=cluster_labels_agg, cmap='viridis')
    ax_tsne_agg.set_title("Clusters hiérarchiques projetés via t-SNE")
    ax_tsne_agg.set_xlabel("t-SNE 1")
    ax_tsne_agg.set_ylabel("t-SNE 2")
    st.pyplot(fig_tsne_agg)

    st.markdown("### Recherche du nombre optimal de cluster (Silhouette Score)")
    scores_agg = []
    for k in range(2, 11):
        cluster_agg_k = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_agg_k = cluster_agg_k.fit_predict(df_scaled)
        score_agg_k = silhouette_score(df_scaled, labels_agg_k)
        scores_agg.append(score_agg_k)
    fig_scores_agg, ax_scores_agg = plt.subplots()
    ax_scores_agg.plot(range(2, 11), scores_agg, marker='o')
    ax_scores_agg.set_xlabel('Nombre de clusters')
    ax_scores_agg.set_ylabel('Silhouette Score')
    ax_scores_agg.set_title('Score de silhouette pour différents nombres de clusters (Agglomératif)')
    st.pyplot(fig_scores_agg)

    st.subheader("DBSCAN")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels_dbscan = dbscan.fit_predict(df_scaled)
    n_clusters_dbscan = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)
    n_noise_dbscan = list(cluster_labels_dbscan).count(-1)
    st.write(f"Nombre de clusters trouvés : {n_clusters_dbscan}")
    st.write(f"Nombre de points considérés comme bruit : {n_noise_dbscan}")

    st.markdown("### Visualisation des clusters à travers PCA")
    pca_dbscan = PCA(n_components=2)
    X_pca_dbscan = pca_dbscan.fit_transform(df_scaled)
    fig_pca_dbscan, ax_pca_dbscan = plt.subplots(figsize=(8, 6))
    ax_pca_dbscan.scatter(X_pca_dbscan[:, 0], X_pca_dbscan[:, 1], c=cluster_labels_dbscan, cmap='viridis', s=50)
    ax_pca_dbscan.set_title('Clustering DBSCAN (projection PCA)' )
    ax_pca_dbscan.set_xlabel('PC 1')
    ax_pca_dbscan.set_ylabel('PC 2')
    st.pyplot(fig_pca_dbscan)

    st.markdown("### Visualisation des clusters à travers t-SNE")
    tsne_dbscan = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne_dbscan = tsne_dbscan.fit_transform(df_scaled)
    fig_tsne_dbscan, ax_tsne_dbscan = plt.subplots(figsize=(8, 5))
    ax_tsne_dbscan.scatter(X_tsne_dbscan[:, 0], X_tsne_dbscan[:, 1], c=cluster_labels_dbscan, cmap='viridis')
    ax_tsne_dbscan.set_title("Clusters DBSCAN projetés via t-SNE")
    ax_tsne_dbscan.set_xlabel("t-SNE 1")
    ax_tsne_dbscan.set_ylabel("t-SNE 2")
    st.pyplot(fig_tsne_dbscan)

    st.subheader("Algorithme K-Means")
    k_kmeans = 6
    kmeans_model = KMeans(n_clusters=k_kmeans, n_init=10, random_state=42)
    cluster_labels_kmeans = kmeans_model.fit_predict(df_scaled)

    st.markdown("### Visualisation des clusters à travers PCA")
    pca_kmeans = PCA(n_components=2)
    X_pca_kmeans = pca_kmeans.fit_transform(df_scaled)
    fig_pca_kmeans, ax_pca_kmeans = plt.subplots(figsize=(8, 6))
    ax_pca_kmeans.scatter(X_pca_kmeans[:, 0], X_pca_kmeans[:, 1], c=cluster_labels_kmeans, cmap='viridis', s=50)
    ax_pca_kmeans.set_title(f"K-Means Clustering (k={k_kmeans}) - Projection PCA")
    ax_pca_kmeans.set_xlabel("PC 1")
    ax_pca_kmeans.set_ylabel("PC 2")
    st.pyplot(fig_pca_kmeans)

    st.markdown("### Visualisation des clusters à travers t-SNE")
    tsne_kmeans = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne_kmeans = tsne_kmeans.fit_transform(df_scaled)
    fig_tsne_kmeans, ax_tsne_kmeans = plt.subplots(figsize=(8, 5))
    ax_tsne_kmeans.scatter(X_tsne_kmeans[:, 0], X_tsne_kmeans[:, 1], c=cluster_labels_kmeans, cmap='viridis')
    ax_tsne_kmeans.set_title("Clusters K-Means projetés via t-SNE")
    ax_tsne_kmeans.set_xlabel("t-SNE 1")
    ax_tsne_kmeans.set_ylabel("t-SNE 2")
    st.pyplot(fig_tsne_kmeans)

    st.markdown("### Recherche du nombre optimal de cluster (Silhouette Score)")
    k_values_kmeans = range(2, 11)
    silhouette_scores_kmeans = []
    for k in k_values_kmeans:
        kmeans_k = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_kmeans_k = kmeans_k.fit_predict(df_scaled)
        score_kmeans_k = silhouette_score(df_scaled, labels_kmeans_k)
        silhouette_scores_kmeans.append(score_kmeans_k)
    fig_scores_kmeans, ax_scores_kmeans = plt.subplots()
    ax_scores_kmeans.plot(k_values_kmeans, silhouette_scores_kmeans, marker='o')
    ax_scores_kmeans.set_title("Silhouette Score vs. Nombre de clusters k (K-Means)")
    ax_scores_kmeans.set_xlabel("Nombre de clusters k")
    ax_scores_kmeans.set_ylabel("Silhouette Score")
    ax_scores_kmeans.grid(True)
    st.pyplot(fig_scores_kmeans)

    st.subheader("Analyse des Personas")
    best_k = 6
    kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["Cluster"] = kmeans_final.fit_predict(df_scaled)

    personas_df = df.groupby('Cluster').agg({
        'Age': ['mean', 'min', 'max'],
        'Annual Income (k$)': ['mean', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'min', 'max'],
        'Cluster': 'count'
    })
    personas_df.columns = ['_'.join(col).strip() for col in personas_df.columns.values]
    personas_df = personas_df.rename(columns={'Cluster_count': 'Effectif'}).reset_index()
    st.write("Analyse des personas :")
    st.dataframe(personas_df)

    st.markdown("### Représentation des proportions des clusters par sexe")
    personas_dict = {
        0: "Seniors stables",
        1: "Jeunes actifs équilibrés",
        2: "Adultes riches économes",
        3: "Jeunes aisés dépensiers",
        4: "Jeunes modestes impulsifs",
        5: "Adultes modestes économes"
    }
    df["ClusterLabel"] = df["Cluster"].map(personas_dict)
    cluster_gender = df.groupby(["ClusterLabel", "Gender"]).size().reset_index(name='count')
    cluster_gender["proportion"] = cluster_gender.groupby('ClusterLabel')['count'].transform(lambda x: x / x.sum())
    fig_gender, ax_gender = plt.subplots(figsize=(12, 7))
    barplot_gender = sns.barplot(
        data=cluster_gender,
        x='ClusterLabel',
        y='proportion',
        hue='Gender',
        palette='Set2',
        ax=ax_gender
    )
    for container in barplot_gender.containers:
        barplot_gender.bar_label(container, labels=[f'{w:.0%}' for w in container.datavalues], label_type='center', fontsize=10, color='black')
    ax_gender.set_title('Proportion Hommes/Femmes par cluster')
    ax_gender.set_xlabel('clusters')
    ax_gender.set_ylabel('Proportion')
    ax_gender.set_ylim(0, 1)
    ax_gender.tick_params(axis='x', rotation=15)
    ax_gender.legend(title='Sexe')
    plt.tight_layout()
    st.pyplot(fig_gender)

    st.markdown("### Représentation 3D des clusters (KMeans)")
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    colors = sns.color_palette("hls", best_k)
    for i in range(best_k):
        cluster_data = df[df["Cluster"] == i]
        ax_3d.scatter(
            cluster_data["Age"],
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            label=f"Cluster {i}: {personas_dict[i]}",
            color=colors[i],
            s=50
        )
    ax_3d.set_xlabel("Âge")
    ax_3d.set_ylabel("Revenu annuel (k$)")
    ax_3d.set_zlabel("Score de dépense (1-100)")
    ax_3d.set_title("Représentation 3D des clusters KMeans (k=6)")
    ax_3d.legend()
    plt.tight_layout()
    st.pyplot(fig_3d)


