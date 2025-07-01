import pandas as pd
import numpy as np
from pathlib import Path
import sys

def fail(msg):
    print(f"{msg}")
    sys.exit(1)

def check_file_exists(path):
    if not path.exists() or path.stat().st_size == 0:
        fail(f"File missing or empty: {path}")
    else:
        print(f"File found: {path}")

def validate_pipeline():
    base = Path("data/processed")

    files = {
        "Heuristic features": base / "psych_features.csv",
        "Embedding IDs": base / "embedding_ids.csv",
        "Embeddings": base / "embeddings.npy",
        "Cluster labels": base / "cluster_labels.csv",
        "Merged dataset": base / "full_features.csv"
    }

    for name, path in files.items():
        check_file_exists(path)

    # Load files
    psych = pd.read_csv(files["Heuristic features"])
    ids = pd.read_csv(files["Embedding IDs"])
    clusters = pd.read_csv(files["Cluster labels"])
    embeddings = np.load(files["Embeddings"])
    full = pd.read_csv(files["Merged dataset"])

    # Check row counts
    if not (len(psych) == len(ids) == len(clusters) == embeddings.shape[0]):
        fail("Mismatch in row counts between psych, IDs, clusters, or embeddings.")
    print("All component datasets have matching row counts.")

    # Check ID consistency
    if not (psych["id"].equals(ids["id"]) and psych["id"].equals(clusters["id"])):
        fail("Mismatch in post IDs across datasets.")
    print("All IDs are aligned across files.")

    # Check full_features shape
    expected_cols = psych.shape[1] + 1 + embeddings.shape[1]  # +1 for cluster
    if full.shape[1] != expected_cols:
        fail(f"full_features.csv has unexpected number of columns ({full.shape[1]} vs expected {expected_cols})")
    print("full_features.csv has correct number of columns.")

    # Check for NaNs
    if full.isnull().any().any():
        fail("NaN values detected in full_features.csv")
    print("No NaNs in full_features.csv")

    # Cluster sanity check
    cluster_counts = clusters["cluster"].value_counts(dropna=False)
    print("Cluster label distribution:")
    print(cluster_counts)

    print("\nPipeline validation complete. All systems nominal.")

if __name__ == "__main__":
    validate_pipeline()
