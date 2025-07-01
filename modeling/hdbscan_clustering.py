import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory path for saving images
images_dir = '/Users/am/python_code/project_folder/standalone_complex_profiler/visualisation/images'

# Create the directory if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Load UMAP-reduced data
df = pd.read_csv('/Users/am/python_code/project_folder/standalone_complex_profiler/data/processed/reddit_with_umap.csv')

# Fit HDBSCAN clustering on 2D UMAP embeddings
clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=15)
clusters = clusterer.fit_predict(df[['umap_x', 'umap_y']])

# Add clustering outputs to DataFrame
df['cluster'] = clusters
df['cluster_probability'] = clusterer.probabilities_  # Confidence score for cluster membership
df['outlier_score'] = clusterer.outlier_scores_       # Outlier score (anomaly detection)

# Generate and print cluster distribution summary
summary = df['cluster'].value_counts().sort_index()
print("Cluster distribution:\n", summary)

# Save DataFrame with HDBSCAN outputs to a new file
df.to_csv('/Users/am/python_code/project_folder/standalone_complex_profiler/data/processed/reddit_with_hdbscan.csv', index=False)
print("Clustering complete and saved to 'data/processed/reddit_with_hdbscan.csv'")

# Visualize clusters on UMAP projection (great for understanding the general structure of your data and seeing the groups formed by HDBSCAN)
plt.figure(figsize=(8, 6))
plt.scatter(df['umap_x'], df['umap_y'], c=df['cluster'], cmap='tab10', s=5)
plt.title('HDBSCAN Clusters on UMAP Projection')
# Save the plot to the images directory
plt.savefig(os.path.join(images_dir, 'umap_clusters.png'))
plt.show()

# Visualize clusters using cluster probabilities (color-coded by probability) if you care about the confidence of the cluster assignments
plt.figure(figsize=(8, 6))
plt.scatter(df['umap_x'], df['umap_y'], c=df['cluster_probability'], cmap='viridis', s=5)
plt.title('HDBSCAN Clusters with Cluster Probability')
plt.colorbar(label='Cluster Probability')  # Add a color bar to show the scale
# Save the plot to the images directory
plt.savefig(os.path.join(images_dir, 'umap_clusters_with_probability.png'))
plt.show()

# Visualize outliers based on outlier scores (color-coded by outlier score) specifically interested in detecting anomalous data points that don't fit into the general structure of the clusters. 
plt.figure(figsize=(8, 6))
plt.scatter(df['umap_x'], df['umap_y'], c=df['outlier_score'], cmap='coolwarm', s=5)
plt.title('Outlier Scores on UMAP Projection')
plt.colorbar(label='Outlier Score')  # Add a color bar to show the scale
# Save the plot to the images directory
plt.savefig(os.path.join(images_dir, 'umap_outlier_scores.png'))
plt.show()
