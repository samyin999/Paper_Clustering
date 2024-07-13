import numpy as np
import os
from preprocess import process_papers
from vectorize import vectorize_papers
from cluster import cluster_papers, elbow_method
from visualize import visualize_clusters
from sklearn.feature_extraction.text import TfidfVectorizer

def get_top_terms(vector_matrix, vectorizer, n_top_terms=5):
    feature_names = vectorizer.get_feature_names_out()
    top_terms = []
    for cluster_center in vector_matrix:
        top_term_indices = cluster_center.argsort()[-n_top_terms:][::-1]
        top_terms.append([feature_names[i] for i in top_term_indices])
    return top_terms

def save_results(clusters, paper_titles, top_terms):
    with open('clustering_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Number of clusters: {len(set(clusters))}\n\n")
        for cluster_id in sorted(set(clusters)):
            cluster_papers = [title for i, title in enumerate(paper_titles) if clusters[i] == cluster_id]
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Number of papers: {len(cluster_papers)}\n")
            if cluster_id >= 0 and cluster_id < len(top_terms):
                f.write(f"  Top terms: {', '.join(top_terms[cluster_id])}\n")
            else:
                f.write("  Top terms: N/A\n")
            f.write("  Papers:\n")
            for paper in cluster_papers:
                f.write(f"    - {paper}\n")
            f.write("\n")

# Preprocess papers
paper_directory = './papers/'
processed_papers = process_papers(paper_directory)
paper_titles = [file for file in os.listdir(paper_directory) if file.endswith('.pdf')]

# Vectorize papers
vector_matrix, vectorizer = vectorize_papers(processed_papers, method='tfidf')

# Determine optimal number of clusters
elbow_method(vector_matrix)

# Perform clustering
n_clusters = 5  # Choose based on elbow method results
clusters, silhouette, davies_bouldin, model = cluster_papers(vector_matrix, method='kmeans', n_clusters=n_clusters)

# Convert clusters to a list if it's not already
clusters = list(clusters)

# Print cluster information
unique_clusters = sorted(set(clusters))
for cluster in unique_clusters:
    count = clusters.count(cluster)
    print(f"Cluster {cluster}: {count} papers")

print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)

# Visualize clusters
visualize_clusters(vector_matrix, clusters, paper_titles, method='tsne')
visualize_clusters(vector_matrix, clusters, paper_titles, method='pca')

# Get top terms for each cluster
if hasattr(model, 'cluster_centers_'):
    top_terms = get_top_terms(model.cluster_centers_, vectorizer)
else:
    top_terms = [[] for _ in range(max(clusters) + 1)]  # Empty list for each cluster, including potential -1

# Save results
save_results(clusters, paper_titles, top_terms)

print("Results saved to clustering_results.txt")
print("Visualizations saved as cluster_visualization_tsne.png and cluster_visualization_pca.png")