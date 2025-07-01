"""
cluster_labels.py

Assigns human-interpretable labels and traits to HDBSCAN clusters based on extracted psychological signals.
Designed for modular, production-grade expansion.

No toy academic toybox but real psycho-symbolic machine 
- sharp and lethal; hopefully won't offend to many social scientists
"""

import pandas as pd
import os
from collections import Counter
import yaml

# Configurable Paths
CLUSTER_DATA_PATH = './data/processed/clustered_data.csv'  # Adjust if necessary
OUTPUT_DIR = './output/cluster_labels/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cluster_data(path: str) -> pd.DataFrame:
    """Load clustered data from CSV."""
    df = pd.read_csv(path)
    if 'cluster' not in df.columns:
        raise ValueError("Cluster column missing in dataset.")
    return df

def extract_dominant_traits(df: pd.DataFrame, cluster_id: int) -> dict:
    """Extract dominant psychological features for a given cluster."""
    subset = df[df['cluster'] == cluster_id]
    traits = {}
    
    if 'valence' in subset.columns:
        traits['avg_valence'] = subset['valence'].mean()
    if 'complexity' in subset.columns:
        traits['avg_complexity'] = subset['complexity'].mean()
    if 'detected_defenses' in subset.columns:
        defenses = sum(subset['detected_defenses'].dropna().apply(eval).tolist(), [])
        traits['common_defenses'] = Counter(defenses).most_common(5)

    return traits

def select_representative_posts(df: pd.DataFrame, cluster_id: int, n_posts: int = 3) -> list:
    """Select top-N posts based on extremity of psychological signals."""
    subset = df[df['cluster'] == cluster_id]
    if subset.empty:
        return []

    subset['signal_score'] = subset[['valence', 'complexity']].abs().sum(axis=1)
    top_posts = subset.nlargest(n_posts, 'signal_score')

    samples = []
    for _, row in top_posts.iterrows():
        samples.append({
            'text': row['text'][:500],  # Clip long posts for readability
            'valence': row.get('valence', None),
            'complexity': row.get('complexity', None),
            'detected_defenses': row.get('detected_defenses', None)
        })
    return samples

def save_cluster_label_yaml(cluster_id: int, label: str, dominant_traits: dict, sample_posts: list):
    """Save cluster label and traits as YAML file."""
    data = {
        'cluster_id': cluster_id,
        'label': label,
        'dominant_traits': dominant_traits,
        'sample_posts': sample_posts
    }
    
    with open(os.path.join(OUTPUT_DIR, f'cluster_{cluster_id}_label.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def manual_labeling_prompt(cluster_id: int, dominant_traits: dict, sample_posts: list) -> str:
    """Ask human to assign a label manually via console prompt."""
    print(f"\n\n===== Cluster {cluster_id} =====")
    print("Dominant Traits:", dominant_traits)
    print("Representative Posts:")
    for idx, post in enumerate(sample_posts, 1):
        print(f"[{idx}] {post['text'][:200]}...\n")

    label = input("Enter label for this cluster: ")
    return label

def main():
    df = load_cluster_data(CLUSTER_DATA_PATH)
    cluster_ids = sorted(df['cluster'].dropna().unique())

    for cluster_id in cluster_ids:
        dominant_traits = extract_dominant_traits(df, cluster_id)
        sample_posts = select_representative_posts(df, cluster_id)
        label = manual_labeling_prompt(cluster_id, dominant_traits, sample_posts)
        save_cluster_label_yaml(cluster_id, label, dominant_traits, sample_posts)

if __name__ == "__main__":
    main()
