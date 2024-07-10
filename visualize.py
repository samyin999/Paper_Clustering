import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def reduce_dimensions(vector_matrix, method='tsne'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method")
    
    return reducer.fit_transform(vector_matrix.toarray() if hasattr(vector_matrix, 'toarray') else vector_matrix)

def visualize_clusters(vector_matrix, clusters, titles, method='tsne'):
    reduced_vectors = reduce_dimensions(vector_matrix, method)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    
    for i, title in enumerate(titles):
        plt.annotate(title, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
    
    plt.title(f'Cluster Visualization using {method.upper()}')
    plt.tight_layout()
    plt.savefig(f'cluster_visualization_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()