from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def apply_kmeans(features, k):
    """Apply K-Means clustering."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(features)
    return labels, model

def evaluate_clustering(labels, features):
    """Evaluate clustering using silhouette score."""
    return silhouette_score(features, labels)
