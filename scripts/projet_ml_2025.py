import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    pairwise_distances
)
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import warnings
from time import time
from itertools import product
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
import dill


file_path = "data/Mall_Customers.csv"

# Chargement des données
df = pd.read_csv(file_path)

# I.1 Aperçu de la base
df.head()

# I.2 Informations d'ordre générales sur la base
print(df.info())

# Vérification de presence de valeurs manquantes
print(df.isnull().sum())

# Affichage des statistiques descriptives des données
df.describe().round(4)

# I.3 Analyse descriptive univariée

## I.3.1  Avec la variable sexe (Gender)

## Nombre d'occurrences de chaque modalité dans la variable 'Gender'
gender_counts = df['Gender'].value_counts()

## Créer un diagramme circulaire en forme de donut
plt.figure(figsize=(8, 6))
plt.pie(gender_counts,
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightblue', 'pink'],
        wedgeprops={'width':0.3},      # Ajouter un trou au centre
        pctdistance=0.80)              # Positionner le % à l'intérieur de la bande

plt.title("Répartition des clients par sexe")
plt.axis('equal')  # Assure un cercle parfait
plt.legend(gender_counts.index, loc='center left', bbox_to_anchor=(1, 0.5), title='Gender')
plt.show()


# I.3.2   Avec la variable "Age"
# Construction de l'histogramme pour la variable "Age"
plt.figure(figsize=(10, 6))
sns.set_style("white")
df['Age'].plot.hist(color='gray', bins=10)
plt.xlabel('Âge')
plt.title('Distribution de l\'âge')
plt.show()

#I.3.3 Avec la variable revenu annuel (Annual Income)

# Construction de l'histogramme pour la variable "Age"
plt.figure(figsize=(10, 6))
sns.set_style("white")
df['Annual Income (k$)'].plot.hist(color='lightblue', bins=10)
plt.xlabel('Annual Income (k$)')
plt.title('Distribution du revenu annuel')
plt.show()

# I.3.3 Avec la variable score de dépenses (Annual Income)

# Histogramme pour la variable 'Spending Score (1-100)'
plt.figure(figsize=(10, 6))
df['Spending Score (1-100)'].plot.hist(color='#6A5ACD', bins=10)  # Choisir un nombre de bins
plt.xlabel('Spending Score (1-100)')
plt.title('Distribution de la variable Spending Score')
plt.show()

# I.3.4 Histogrammes et courbes de densité

#  Histogrammes et courbes de densité
plt.figure(figsize=(15, 5))

# Histogramme pour les variables "Age"
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title('Histogramme de l\'Âge')
plt.xlabel('Âge')

# Histogramme pour 'Annual Income (k$)'
plt.subplot(1, 3, 2)
sns.histplot(df['Annual Income (k$)'], kde=True, color='lightgreen')
plt.title('Histogramme du Revenu Annuel')
plt.xlabel('Revenu Annuel (k$)')

# Histogramme pour 'Spending Score (1-100)'
plt.subplot(1, 3, 3)
sns.histplot(df['Spending Score (1-100)'], kde=True, color='salmon')
plt.title('Histogramme du Score de Dépenses')
plt.xlabel('Score de Dépenses')

plt.tight_layout()
plt.show()


## I.4 Analyses descriptives bivariées

## I.4.1 Distribution par sexe

# Définir le style de la visualisation
sns.set_style("white")

# Créer la figure
plt.figure(figsize=(10, 6))

# Créer un boxplot pour la variable "Spending Score" selon le sexe
plt.boxplot([df[df['Gender'] == 'Male']['Spending Score (1-100)'],
             df[df['Gender'] == 'Female']['Spending Score (1-100)']])

plt.xticks([1, 2], ['Male', 'Female']) # Ajout de labels pour l'axe X
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.title('Gender vs Spending Score')
plt.show()


### I.4.2 Test de Mann-Whitney-Wilcoxon (aussi appelé Wilcoxon rank-sum test)


male_scores = df[df['Gender'] == 'Male']['Spending Score (1-100)']
female_scores = df[df['Gender'] == 'Female']['Spending Score (1-100)']

# Test bilatéral (alternative='two-sided')
stat, p_value = mannwhitneyu(male_scores, female_scores, alternative='two-sided')

print(f"Statistique de test : {stat}")
print(f"Valeur p : {p_value}")


# Pairplot pour "Age", "Annual Income (k$)" et "Spending Score (1-100)"
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle('Pairplot des Variables', y=1.02)
plt.show()


## I.4.4 Matrice de correlation

# Calcul de la matrice de corrélation
corr_matrix = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()

# Heatmap de la matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()


## I.5 Standardisation des données
scaler = StandardScaler()

cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_scaled = df.copy()

df_scaled[cols_to_scale] = scaler.fit_transform\
(df[cols_to_scale])

df_scaled[cols_to_scale].describe()

df_scaled=df_scaled[cols_to_scale]
df_scaled.head(3)


## II.1. Clustering Hiérarchique (Agglomératif)

### II.1.1 Dendrogram pour avoir une idée des cluster possible

plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(df_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], method='ward'))
plt.title("Dendrogramme (Méthode de Ward)")
plt.xlabel("Indices des Clients")
plt.ylabel("Distance Euclidienne")
plt.show()

### II.1.2 Dendrogram pour representer 20 clusters

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
Z = linkage(df_scaled[['Age','Annual Income (k$)', 'Spending Score (1-100)']], 'ward') # use ward linkage
dn = dendrogram(Z, truncate_mode='lastp', p=20) # show last 10 clusters
plt.show()


### II.1.2 Visualisation des clusters à travers la reduction de dimension PCA
# Clustering hiérarchique
clustering = AgglomerativeClustering(n_clusters=6)  # choisis le nombre de clusters selon ton cas
cluster_labels = clustering.fit_predict(df_scaled)

# PCA pour visualisation 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

# Affichage
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Clusters hiérarchiques projetés via PCA")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.colorbar()
plt.show()


### II.1.3 Visualisation des clusters à travers TSNE

# Clustering hiérarchique
clustering = AgglomerativeClustering(n_clusters=6)  # choisir le nombre de clusters selon ton cas
cluster_labels = clustering.fit_predict(df_scaled)

# t-SNE pour visualisation 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(df_scaled)

# Affichage
plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Clusters hiérarchiques projetés via t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.show()


### II.1.4 Recherche du nombre optimal de cluster à travers un algo

scores = []
for k in range(2, 11):
    cluster = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = cluster.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    scores.append(score)

# Tracé du score en fonction du nombre de clusters
plt.plot(range(2, 11), scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette Score')
plt.title('Score de silhouette pour différents nombres de clusters')
plt.show()


### II.1.5 Evaluation des performances du modèle

# 3. Évaluation des performances
silhouette = silhouette_score(df_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(df_scaled, cluster_labels)

# 4. Affichage
print(" Évaluation du clustering hiérarchique")
print(f"- Silhouette Score        : {silhouette:.3f} (proche de 1 = bien)")
print(f"- Davies-Bouldin Index    : {davies_bouldin:.3f} (proche de 0 = bien)")
print(f"- Calinski-Harabasz Score : {calinski_harabasz:.3f} (plus élevé = bien)")

## II.2. DBSCAN

# 2. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Ajuste eps si besoin
cluster_labels = dbscan.fit_predict(df_scaled)

# 3. Affichage du nombre de clusters trouvés (hors bruit)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Nombre de clusters trouvés : {n_clusters}")
print(f"Nombre de points considérés comme bruit : {n_noise}")

## II.2.1 Visualisation des clusters à travers la reduction de dimension PCA

# 5. Visualisation (option : PCA pour projeter)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('Clustering DBSCAN (projection PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar(label='Cluster')
plt.show()


### II.2.2 Visualisation des clusters à travers TSNE

# t-SNE pour visualisation 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(df_scaled)

# Affichage
plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Clusters hiérarchiques projetés via t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.show()


## II.2.3 Evaluation des performances du modèle

# 4. Évaluation (uniquement si au moins 2 clusters)
if n_clusters > 1:
    silhouette = silhouette_score(df_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(df_scaled, cluster_labels)

    print("\n Évaluation du clustering DBSCAN")
    print(f"- Silhouette Score        : {silhouette:.3f}")
    print(f"- Davies-Bouldin Index    : {davies_bouldin:.3f}")
    print(f"- Calinski-Harabasz Score : {calinski_harabasz:.3f}")
else:
    print(" DBSCAN a détecté moins de 2 clusters. Évaluation non applicable.")

## II.3 L'algorithme du K-means

warnings.filterwarnings("ignore", category=UserWarning)
# 2. K-Means clustering
k = 6  # Choisis le nombre de clusters ici
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

### II.3.1 Visualisation des clusters à travers la reduction de dimension PCA"""

warnings.filterwarnings("ignore", category=UserWarning)



# 4. Visualisation avec PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title(f"K-Means Clustering (k={k}) - Projection PCA")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar(label='Cluster')
plt.show()


### II.3.2 Visualisation des clusters à travers TSNE

# Clustering hiérarchique
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

# t-SNE pour visualisation 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(df_scaled)

# Affichage
plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Clusters hiérarchiques projetés via t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.show()

### II.3.3 Recherche du nombre optimal de cluster à travers un algo

warnings.filterwarnings("ignore", category=UserWarning)

# 2. Paramètres de la boucle
k_values = range(2, 11)
silhouette_scores = []

# 3. Boucle sur k
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(df_scaled)

    # Silhouette Score
    score = silhouette_score(df_scaled, labels)
    silhouette_scores.append(score)

# 4. Affichage du Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Nombre de clusters k")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


## II.3.4 Evaluation des performances du modèle
# 3. Évaluation du modèle
k=6
silhouette = silhouette_score(df_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(df_scaled, cluster_labels)

print(f" K-Means avec k = {k}")
print(f"- Silhouette Score        : {silhouette:.3f}")
print(f"- Davies-Bouldin Index    : {davies_bouldin:.3f}")
print(f"- Calinski-Harabasz Score : {calinski_harabasz:.3f}")


#  III. Optimisation des hyperparamètres du meilleurs modèle


# 1. Préparation des données
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)

# 2. Meilleur nombre de clusters (déjà trouvé)
# 3. Paramètres à tester
best_k = 6  # Remplace par ton meilleur k si différent

init_options = ['k-means++', 'random']
n_init_options = [10,20]
max_iter_options = [300]
algorithm_options = ['elkan']

# 4. Stocker les résultats
results = []

# 5. Grid search manuel
for init, n_init, max_iter, algo in product(init_options, n_init_options, max_iter_options, algorithm_options):
    try:
        model = KMeans(
            n_clusters=best_k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algo,
            random_state=42
        )
        labels = model.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)

        results.append({
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'algorithm': algo,
            'Silhouette Score': round(silhouette, 4)
        })
    except Exception as e:
        print(f" Erreur avec {init}, {n_init}, {max_iter}, {algo} : {e}")

# 6. Affichage des meilleurs résultats
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Silhouette Score', ascending=False)

print("\n Top 10 meilleures combinaisons :")
print(results_df.head(10).to_string(index=False))

# IV. Analyse des personnas

## IV.1 Formation et analyse des clusters

# 1. Sélection des variables
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 2. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering avec KMeans (k = 3 ou le meilleur nombre trouvé)
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Analyse des personas
personas = df.groupby('Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'Cluster': 'count'  # pour compter le nombre d'individus
})

# 5. Nettoyage des noms de colonnes
personas.columns = ['_'.join(col).strip() for col in personas.columns.values]
personas = personas.rename(columns={'Cluster_count': 'Effectif'}).reset_index()

# 6. Affichage des résultats
print(" Analyse des personas :")
print(personas)

## IV.2 Representation de proportions des cluster par sexe

# 1. Dictionnaire des personas
personas = {
    0: "Seniors stables",
    1: "Jeunes actifs équilibrés",
    2: "Adultes riches économes",
    3: "Jeunes aisés dépensiers",
    4: "Jeunes modestes impulsifs",
    5: "Adultes modestes économes"
}

# 2. Remplacer les numéros de cluster par les labels dans le DataFrame
df['ClusterLabel'] = df['Cluster'].map(personas)

# 3. Regroupement par Cluster et Genre
cluster_gender = df.groupby(['ClusterLabel', 'Gender']).size().reset_index(name='count')

# 4. Calcul des proportions
cluster_gender['proportion'] = cluster_gender.groupby('ClusterLabel')['count'].transform(lambda x: x / x.sum())

# 5. Affichage du graphique
plt.figure(figsize=(12, 7))
barplot = sns.barplot(
    data=cluster_gender,
    x='ClusterLabel',
    y='proportion',
    hue='Gender',
    palette='Set2'
)

# 6. Ajouter les pourcentages sur les barres
for container in barplot.containers:
    barplot.bar_label(container, labels=[f'{w:.0%}' for w in container.datavalues], label_type='center', fontsize=10, color='black')

# 7. Mise en forme
plt.title('Proportion Hommes/Femmes par cluster')
plt.xlabel('clusters')
plt.ylabel('Proportion')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title='Sexe')
plt.tight_layout()
plt.show()

# V. Répresentation en 3D des clusters à travers le meilleur modèle Kmeans</span>"""

# 1. Réentraîner le meilleur modèle avec les meilleurs hyperparamètres trouvés
best_params = results_df.iloc[0]  # Première ligne du DataFrame trié

best_model = KMeans(
    n_clusters=best_k,
    init=best_params['init'],
    n_init=int(best_params['n_init']),
    max_iter=int(best_params['max_iter']),
    algorithm=best_params['algorithm'],
    random_state=42
)

df['Cluster'] = best_model.fit_predict(X_scaled)

# 2. Dictionnaire des personas
personas = {
    0: "Seniors stables",
    1: "Jeunes actifs équilibrés",
    2: "Adultes riches économes",
    3: "Jeunes aisés dépensiers",
    4: "Jeunes modestes impulsifs",
    5: "Adultes modestes économes"
}

df['Persona'] = df['Cluster'].map(personas)

# 3. Affichage 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Couleurs
colors = sns.color_palette("hls", best_k)

for i in range(best_k):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(
        cluster_data['Age'],
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        label=f"Cluster {i}: {personas[i]}",
        color=colors[i],
        s=50
    )

ax.set_xlabel("Âge")
ax.set_ylabel("Revenu annuel (k$)")
ax.set_zlabel("Score de dépense (1-100)")
ax.set_title("Représentation 3D des clusters KMeans (k=6)")
ax.legend()
plt.tight_layout()
plt.show()

joblib.dump({
    "model": kmeans,
    "scaler": scaler,
    "features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
}, "models/kmeans_model.pkl")
