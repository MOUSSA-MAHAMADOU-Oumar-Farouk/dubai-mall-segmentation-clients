
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings

from src.data_preprocessing import load_and_preprocess_data

warnings.filterwarnings("ignore", category=UserWarning)

def train_and_save_model(file_path="data/Mall_Customers.csv", model_path="models/kmeans_model.pkl"):
    df, scaler = load_and_preprocess_data(file_path)

    # Select only numerical columns for clustering after preprocessing
    cols_for_clustering = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled = df[cols_for_clustering]

    # Find optimal number of clusters using Silhouette Score
    k_values = range(2, 11)
    best_silhouette_score = -1
    best_k = 0
    best_kmeans_model = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_k = k
            best_kmeans_model = kmeans

    print(f"Meilleur nombre de clusters trouvé : {best_k} avec un Silhouette Score de {best_silhouette_score:.3f}")

    # Train the final model with the best k
    final_kmeans_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    final_kmeans_model.fit(X_scaled)

    # Save the model and the scaler
    joblib.dump({
        'model': final_kmeans_model,
        'scaler': scaler,
        'features': cols_for_clustering
    }, model_path)
    print(f"Modèle K-Means et scaler sauvegardés dans {model_path}")

if __name__ == "__main__":
    train_and_save_model()


