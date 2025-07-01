# Layer 2: Domain-Adaptive Sentiment
# like a neural upgrade, think of this as giving the feature extractor night vision for emotional nuance.
import pandas as pd
import numpy as np
import argparse
import re
from textblob import TextBlob
from transformers import pipeline
from pathlib import Path

# --- Setup RoBERTa sentiment model ---
roberta_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# --- Psych feature functions ---
def count_i(text):
    return len(re.findall(r"\b[Ii]\b", text))

def count_negations(text):
    return len(re.findall(r"\b(not|no|never|n't)\b", text))

def count_questions(text):
    return text.count("?")

def count_temporal(text):
    return len(re.findall(r"\b(yesterday|today|tomorrow|week|month|year|day|decade)\b", text))

def get_blob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def get_roberta_scores(text):
    result = roberta_pipe(text[:512])[0]  # Truncate to 512 tokens
    scores = {
        "roberta_sent_neg": result["score"] if result["label"] == "LABEL_0" else 0,
        "roberta_sent_neu": result["score"] if result["label"] == "LABEL_1" else 0,
        "roberta_sent_pos": result["score"] if result["label"] == "LABEL_2" else 0,
    }
    return scores

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Extract psychological features from Reddit text")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned input CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    features = {
        "id": [],
        "word_count": [],
        "i_count": [],
        "negation_count": [],
        "question_mark_count": [],
        "temporal_refs": [],
        "sentiment_polarity": [],
        "sentiment_subjectivity": [],
        "roberta_sent_neg": [],
        "roberta_sent_neu": [],
        "roberta_sent_pos": []
    }

    for _, row in df.iterrows():
        text = (row.get("selftext") or row.get("title") or "")
        features["id"].append(row["id"])
        features["word_count"].append(len(text.split()))
        features["i_count"].append(count_i(text))
        features["negation_count"].append(count_negations(text))
        features["question_mark_count"].append(count_questions(text))
        features["temporal_refs"].append(count_temporal(text))

        polarity, subjectivity = get_blob_sentiment(text)
        features["sentiment_polarity"].append(polarity)
        features["sentiment_subjectivity"].append(subjectivity)

        roberta_scores = get_roberta_scores(text)
        features["roberta_sent_neg"].append(roberta_scores["roberta_sent_neg"])
        features["roberta_sent_neu"].append(roberta_scores["roberta_sent_neu"])
        features["roberta_sent_pos"].append(roberta_scores["roberta_sent_pos"])

    out_df = pd.DataFrame(features)

    # Merge on 'id'
    original_df = pd.read_csv(input_path)
    merged_df = pd.merge(original_df, out_df, on="id", how="left")

    # Save alongside original filename
    output_path = Path("data/processed") / (input_path.stem + "_signals.csv")
    out_df.to_csv(output_path, index=False)
    print(f"Psychological feature file saved to {output_path}")

if __name__ == "__main__":
    main()
