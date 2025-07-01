"""
auto_label.py

LLM-driven automatic cluster labeling for psychological profiling.
Loads clustered posts, extracts top features and representative posts,
queries LLM, and saves results as draft YAML files in outputs/cluster_labels/.
"""

import os
import pandas as pd
import yaml
from collections import Counter
import anthropic
import decimal

# === CONFIG ===
client = anthropic.Anthropic()
CLUSTER_DATA_PATH = './data/processed/reddit_with_clusters_signals_final.csv'
OUTPUT_DIR = './outputs/cluster_labels/'
MODEL_NAME = 'claude-3-opus-20240229'
TOP_N_POSTS = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_for_yaml(obj):
    """Recursively convert Decimals and floats for YAML dumping."""
    if isinstance(obj, dict):
        return {k: clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_yaml(i) for i in obj]
    elif isinstance(obj, (decimal.Decimal, float)):
        return float(obj)
    else:
        return obj

def load_data(path):
    df = pd.read_csv(path)
    if 'cluster' not in df.columns:
        raise ValueError("Cluster column missing.")
    return df

def extract_top_posts(df, cluster_id, n=TOP_N_POSTS):
    subset = df[df['cluster'] == cluster_id].copy()
    subset['signal_score'] = subset['sentiment_polarity_y'].abs()
    top = subset.nlargest(n, 'signal_score')
    return [
        {
            'text': row['selftext'][:500],
            'valence': row.get('sentiment_polarity_y'),
            'complexity': row.get('sentiment_subjectivity_y'),
            'detected_defenses': None
        } for _, row in top.iterrows()
    ]

def summarize_traits(df, cluster_id):
    subset = df[df['cluster'] == cluster_id]
    traits = {
        'avg_sentiment_polarity': subset['sentiment_polarity_y'].mean() if 'sentiment_polarity_y' in subset else None,
        'avg_sentiment_subjectivity': subset['sentiment_subjectivity_y'].mean() if 'sentiment_subjectivity_y' in subset else None,
        'avg_word_count': subset['word_count'].mean() if 'word_count' in subset else None
    }
    return traits

def build_prompt(cluster_id, traits, posts):
    prompt = f"""
You are operating as a clinical meta-analyst tasked with mapping psycho-symbolic complexes at scale.

Given a sample of trauma-saturated online speech and extracted affective-linguistic signals for a specific cluster:
- Synthesize a symbolic label that captures the core dissociative, narcissistic, or existential defenses in play.
- Identify 3-5 dominant psychological features or contradictions.
- Infer, in 1-2 concise sentences, the likely psychological wound and the defensive strategies protecting it.
- Do not focus on surface emotion. Focus on structure: symbolic distortions, defensive adaptations, and narrative anomalies.
- Language must remain clinical, symbolic, and precise — no colloquial sympathy, no therapeutic platitudes.

Data Provided:
- Aggregate Traits: {traits}

Representative Posts:
"""
    for i, post in enumerate(posts):
        prompt += f"\n[{i+1}] {post['text']}\nValence: {post['valence']} | Complexity: {post['complexity']}\nDefenses: {post['detected_defenses']}\n"

    prompt += """
Return strictly:
- Label: (2-4 words, symbolic, archetypal if possible)
- Dominant Traits:
  - (feature 1)
  - (feature 2)
  - (feature 3)
- Inferred Psychological Structure:
  (1-2 sentences, clinical narrative)
"""
    return prompt

def query_llm(prompt):
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            temperature=0.3,
            system="You are a clinical psychological profiler.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude API call failed: {e}")
        raise

def parse_llm_output(llm_text):
    """Parse the raw LLM output into structured label, traits, and structure fields."""

    lines = llm_text.strip().splitlines()
    output = {
        'raw_response': llm_text,
        'label': None,
        'traits': [],
        'structure': ''
    }

    parsing_traits = False
    parsing_structure = False

    for line in lines:
        line = line.strip()

        # Capture Label (whether dashed or not)
        if 'label:' in line.lower():
            label_candidate = line.split(':', 1)[-1].strip()
            if label_candidate:
                output['label'] = label_candidate
            parsing_traits = False
            parsing_structure = False

    for line in lines:
        line = line.strip()

        # Capture Label
        if line.lower().startswith('label:'):
            output['label'] = line.split(':', 1)[-1].strip()
            parsing_traits = False
            parsing_structure = False

        # Detect Dominant Traits Section
        elif line.lower().startswith('dominant traits:'):
            parsing_traits = True
            parsing_structure = False

        # Detect Inferred Psychological Structure Section
        elif line.lower().startswith('inferred psychological structure:'):
            parsing_structure = True
            parsing_traits = False

        # Capture traits
        elif parsing_traits and line.startswith('-'):
            trait = line.lstrip('-').strip()
            if trait:
                output['traits'].append(trait)

        # Capture structure narrative
        elif parsing_structure:
            output['structure'] += ' ' + line.strip()

    # Clean final structure text
    output['structure'] = output['structure'].strip()

    return output

def save_yaml(cluster_id, label_data):
    """Safely save clean minimal YAML for labeling."""
    safe_yaml = {
        'cluster_id': label_data['cluster_id'],
        'label': label_data['label'],
        'traits': label_data['traits'],
        'structure': label_data['structure']
    }

    output_path = os.path.join(OUTPUT_DIR, f'cluster_{cluster_id}_label_draft.yaml')

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(clean_for_yaml(safe_yaml), f, sort_keys=False, allow_unicode=True)
        print(f"[✓] Saved YAML for cluster {cluster_id} at {output_path}")
    except Exception as e:
        print(f"[X] Failed to save YAML for cluster {cluster_id}: {e}")

def main():
    df = load_data(CLUSTER_DATA_PATH)
    cluster_ids = sorted(df['cluster'].dropna().unique())

    cluster_ids = [cid for cid in cluster_ids if cid >= 0]

    for cluster_id in cluster_ids:
        traits = summarize_traits(df, cluster_id)
        posts = extract_top_posts(df, cluster_id)
        prompt = build_prompt(cluster_id, traits, posts)
        print(f"\n--- Cluster {cluster_id} ---\nPrompting LLM...")
        llm_text = query_llm(prompt)

        print("\nClaude raw output:\n", llm_text)

        label_data = parse_llm_output(llm_text)
        label_data['cluster_id'] = int(cluster_id)
        label_data['dominant_traits'] = traits
        label_data['sample_posts'] = posts

        print(label_data)

        save_yaml(cluster_id, label_data)
        print(f"Cluster {cluster_id} labeled as: {label_data['label']}")

if __name__ == '__main__':
    main()
