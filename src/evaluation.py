from sklearn.metrics import silhouette_score

def evaluate_clustering(labels, features):
    score = silhouette_score(features, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score