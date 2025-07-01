# The hybrid feature unifier, the sacred scroll that fuses surface heuristics with deep structure 
# and latent group identity.
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def main():
    print("Searching for latest psychological signals CSV...")
    signal_files = sorted(glob.glob("data/processed/*_signals.csv"))
    if not signal_files:
        raise FileNotFoundError("No *_signals.csv file found in data/processed/")
    latest_signals = signal_files[-1]
    print(f"Found: {latest_signals}")

    print("Loading psych features...")
    heuristics = pd.read_csv(latest_signals)

    print("Loading embeddings...")
    embeddings = np.load("data/processed/embeddings.npy")
    ids = pd.read_csv("data/processed/embedding_ids.csv")

    print("Loading cluster labels...")
    clusters = pd.read_csv("data/processed/cluster_labels.csv")

    print("Merging all components...")
    df = heuristics.merge(ids, on="id").merge(clusters, on="id")

    if embeddings.shape[0] != df.shape[0]:
        raise ValueError(
            f"Embedding count ({embeddings.shape[0]}) does not match merged rows ({df.shape[0]})."
        )

    embedding_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    full_df = pd.concat([df, embedding_df], axis=1)

    output_path = Path("data/processed/full_features.csv")
    full_df.to_csv(output_path, index=False)

    print(f"Merged dataset saved to {output_path} (rows: {full_df.shape[0]})")

if __name__ == "__main__":
    main()

