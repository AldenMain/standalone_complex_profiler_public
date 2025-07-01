import pandas as pd
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    # --- Paths ---
    processed_dir = Path("data/processed")
    input_file = Path("data/raw/OffMyChest_posts_20250418_123424.jsonl")
    print(f"Using input file: {input_file.name}")

    # --- Load full JSONL post data ---
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            post = json.loads(line)
            post["text"] = f"{post['title']} {post['selftext']}"
            records.append(post)

    df = pd.DataFrame(records)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the parsed JSONL.")

    texts = df["text"].tolist()
    ids = df["id"].tolist()

    # --- Load model ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Sentence-BERT model loaded.")

    # --- Generate embeddings ---
    print(f"Encoding {len(texts)} posts...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # --- Save outputs ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "embeddings.npy", embeddings)
    df[["id", "title", "selftext"]].to_csv(processed_dir / "reddit_with_umap.csv", index=False)

    print("Saved embeddings to embeddings.npy")
    print("Saved post metadata to reddit_with_umap.csv")

if __name__ == "__main__":
    main()
