import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict

# === Configuration ===
FINALS_DIR = Path('./outputs/cluster_labels/finals/')
PROFILES_DIR = Path('./outputs/profiles/')

# Create profiles directory if not exists
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# === Utility Functions ===

def load_yaml(path: Path) -> Dict:
    try:
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            print(f"[!] Warning: {path} is empty or invalid. Skipping.")
            return {}
        return data
    except Exception as e:
        print(f"[!] Failed to load YAML {path}: {e}")
        return {}

def save_markdown(content: str, path: Path):
    try:
        with path.open('w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"[!] Failed to save Markdown {path}: {e}")

def assemble_markdown(cluster_data: Dict) -> str:
    cluster_id = cluster_data.get('cluster_id', 'Unknown ID')
    label = cluster_data.get('label', 'No Label')
    traits = cluster_data.get('traits', [])
    structure = cluster_data.get('structure', 'No structure description provided.')
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    traits_formatted = ', '.join(traits) if traits else 'None listed'

    markdown = f"""# Cluster Profile: {label}

**Cluster ID:** {cluster_id}  
**Generated:** {timestamp}

---

## 🔹 Dominant Traits
{traits_formatted}

---

## 🔹 Psychological Structure
{structure}

---

## 🔹 Top Example Posts
(*Placeholder: integrate top posts if available.*)

---

*Profile generated automatically by Standalone Complex Profiler.*
"""
    return markdown

def process_clusters():
    final_files = [f for f in FINALS_DIR.glob('*.yaml')]

    if not final_files:
        print("[!] No final labeled YAML files found.")
        return

    for yaml_path in final_files:
        data = load_yaml(yaml_path)
        if not data:
            continue

        cluster_id = data.get('cluster_id')
        if not cluster_id:
            print(f"[!] Skipping {yaml_path.name}: Missing cluster_id.")
            continue

        profile_markdown = assemble_markdown(data)
        output_path = PROFILES_DIR / f"cluster_{cluster_id}.md"
        save_markdown(profile_markdown, output_path)
        print(f"[✓] Profile exported: {output_path.name}")

def main():
    print("=== Cluster Profile Assembly Started ===")
    process_clusters()
    print("=== Cluster Profile Assembly Complete ===")

if __name__ == "__main__":
    main()
