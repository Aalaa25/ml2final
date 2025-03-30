import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_clusters(tfidf_features, cluster_labels, dataset_name):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(tfidf_features.toarray())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=cluster_labels, palette="viridis")
    plt.title(f"Cluster Visualization for {dataset_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    results_path = "../results"
    os.makedirs(results_path, exist_ok=True)
    
    plot_filename = f"{results_path}/pca_{dataset_name}.png"
    plt.savefig(plot_filename)
    plt.show()
