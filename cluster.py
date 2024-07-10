from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

def kmeans_clustering(vector_matrix, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vector_matrix)
    return clusters, kmeans

def dbscan_clustering(vector_matrix, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(vector_matrix)
    return clusters, dbscan

def evaluate_clustering(vector_matrix, clusters):
    if len(np.unique(clusters)) > 1:
        silhouette = silhouette_score(vector_matrix, clusters)
        # Convert sparse matrix to dense if necessary
        if issparse(vector_matrix):
            vector_matrix = vector_matrix.toarray()
        davies_bouldin = davies_bouldin_score(vector_matrix, clusters)
        return silhouette, davies_bouldin
    else:
        return None, None

def elbow_method(vector_matrix, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vector_matrix)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

def cluster_papers(vector_matrix, method='kmeans', n_clusters=5, eps=0.5, min_samples=5):
    if method == 'kmeans':
        clusters, model = kmeans_clustering(vector_matrix, n_clusters)
    elif method == 'dbscan':
        clusters, model = dbscan_clustering(vector_matrix, eps, min_samples)
    else:
        raise ValueError("Unsupported clustering method")
    
    silhouette, davies_bouldin = evaluate_clustering(vector_matrix, clusters)
    return clusters, silhouette, davies_bouldin, model

if __name__ == "__main__":
    # This is just for testing
    from preprocess import process_papers
    from vectorize import vectorize_papers
    
    paper_directory = './papers/'
    processed_papers = process_papers(paper_directory)
    vector_matrix, _ = vectorize_papers(processed_papers, method='tfidf')
    
    elbow_method(vector_matrix)
    
    clusters, silhouette, davies_bouldin, _ = cluster_papers(vector_matrix, method='kmeans', n_clusters=5)
    
    print("Number of papers in each cluster:", np.bincount(clusters))
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", davies_bouldin)