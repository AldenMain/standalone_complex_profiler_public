import json
import pandas as pd
from pathlib import Path
import argparse
import re

def clean_text(text):
    """Basic text cleaning: remove URLs, newlines, excess whitespace."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.replace("\n", " ").replace("\r", "")
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.strip()

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Clean and convert Reddit JSONL to CSV")
parser.add_argument("--input", type=str, required=True, help="Path to input .jsonl file")
args = parser.parse_args()

# --- Load and clean ---
input_path = Path(args.input)
output_path = Path("data/processed") / input_path.with_suffix(".csv").name
output_path.parent.mkdir(parents=True, exist_ok=True)

data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            post = json.loads(line)
            text = f"{post['title']} {post['selftext']}"
            cleaned = clean_text(text)
            data.append({
                "id": post["id"],
                "subreddit": post["subreddit"],
                "text": cleaned,
                "score": post["score"],
                "num_comments": post["num_comments"]
            })
        except Exception as e:
            print(f"⚠️ Skipping bad line: {e}")

# --- Save as CSV ---
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to {output_path} (rows: {len(df)})")
