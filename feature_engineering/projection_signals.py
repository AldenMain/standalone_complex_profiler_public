# Initial scaffhold - modular, scalable, and ready to plug into the pipeline
# feature_engineering/projection_signals.py

import re
import numpy as np
from textblob import TextBlob
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# --- 1. Pronoun Distance Ratio --- 
# Defensive projection (blame vs ownership)
def pronoun_distance_ratio(text):
    tokens = word_tokenize(text.lower())
    you_they = sum(1 for w in tokens if w in ["you", "they", "them"])
    i_me = sum(1 for w in tokens if w in ["i", "me", "my"])
    return you_they / (i_me + 1)  # Avoid div by zero

# --- 2. Narrative Rigidity Score ---
# Fixed schema / absolutism
def narrative_rigidity_score(text):
    rigidity_terms = ["always", "never", "clearly", "should", "obviously"]
    return sum(text.lower().count(word) for word in rigidity_terms)

# --- 3. Sentiment Valence Variance ---
# Projection - Affective instability (idealization/splitting)
def sentiment_valence_variance(text, chunk_size=3):
    blob = TextBlob(text)
    sentences = blob.sentences
    chunks = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]
    polarity_scores = [np.mean([s.sentiment.polarity for s in chunk]) for chunk in chunks if chunk]
    return np.var(polarity_scores) if len(polarity_scores) > 1 else 0.0

# --- 4. Tense Shifting Score ---
# Temporal dissonance =, unrersolved trauma loop
def detect_tense_shifts(text):
    # Crude regex pass — can upgrade later with SpaCy POS parsing
    past_tense = len(re.findall(r"\b(was|had|did|felt|said|thought)\b", text.lower()))
    present_tense = len(re.findall(r"\b(is|has|do|feel|say|think)\b", text.lower()))
    return abs(past_tense - present_tense)

# --- Master Feature Extractor ---
def extract_projection_features(text):
    return {
        "pronoun_distance_ratio": pronoun_distance_ratio(text),
        "narrative_rigidity_score": narrative_rigidity_score(text),
        "projection_valence_variance": sentiment_valence_variance(text),
        "tense_shifting_score": detect_tense_shifts(text),
    }
