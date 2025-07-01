import numpy as np
import pandas as pd
from pathlib import Path
import umap
import hdbscan
import os

def main():
    # Set up paths
    embeddings_path = Path("data/processed/embeddings.npy")
    ids_path = Path("data/processed/embedding_ids.csv")
    umap_out_path = Path("data/processed/embeddings_umap.npy")
    cluster_out_path = Path("data/processed/cluster_labels.csv")

    # Check if input files exist
    if not embeddings_path.exists() or not ids_path.exists():
        print(f"Missing files! Ensure {embeddings_path} and {ids_path} exist.")
        return

    # Load data
    embeddings = np.load(embeddings_path)
    ids_df = pd.read_csv(ids_path)

    print(f"Loaded {len(embeddings)} embeddings.")

    # Dimensionality reduction with UMAP
    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.1, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    np.save(umap_out_path, embeddings_2d)  # Saving the UMAP output
    print(f"Saved UMAP embeddings to {umap_out_path}")

    # Clustering with HDBSCAN
    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True)
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    probs = clusterer.probabilities_

    # Add to DataFrame
    ids_df["cluster"] = cluster_labels
    ids_df["cluster_prob"] = probs

    # Print number of clusters
    unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN found {unique_clusters} cluster(s)")

    # Save to disk
    ids_df.to_csv(cluster_out_path, index=False)
    print(f"Cluster labels saved to {cluster_out_path}")

if __name__ == "__main__":
    main()
