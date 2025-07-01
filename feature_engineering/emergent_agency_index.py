import pandas as pd
import numpy as np
import re
from textblob import TextBlob

def compute_emergent_agency_index(texts):
    '''
    Compute the Emergent Agency Index (EAI) for a list of text inputs.
    Parameters:
        texts (list of str): List of user-generated texts/posts.
    Returns:
        DataFrame: A DataFrame with calculated EAI components and overall score.
    '''

    data = {
        'text': [],
        'narrative_self_reference': [],
        'ethical_reflection': [],
        'individual_voice_divergence': [],
        'existential_awareness': [],
        'emergent_agency_index': []
    }

    for text in texts:
        # Normalize
        clean_text = text.lower()

        # Narrative Self-Reference Score
        first_person_score = len(re.findall(r'\b(i|my|me|mine)\b', clean_text)) / max(len(clean_text.split()), 1)

        # Ethical Reflection Score
        ethical_keywords = ['right thing', 'wrong', 'should have', 'regret', 'guilt', 'responsible', 'consequence']
        ethical_score = sum(1 for word in ethical_keywords if word in clean_text) / len(ethical_keywords)

        # Individual Voice Divergence (using sentiment complexity as a proxy)
        sentiment = TextBlob(clean_text).sentiment
        sentiment_complexity = abs(sentiment.polarity * sentiment.subjectivity)

        # Existential Awareness Score
        existential_keywords = ['meaning', 'existence', 'purpose', 'death', 'irrelevant', 'ghost', 'identity']
        existential_score = sum(1 for word in existential_keywords if word in clean_text) / len(existential_keywords)

        # Aggregate Emergent Agency Index (weighted)
        eai = (
            0.3 * first_person_score +
            0.25 * ethical_score +
            0.25 * sentiment_complexity +
            0.2 * existential_score
        )

        data['text'].append(text)
        data['narrative_self_reference'].append(first_person_score)
        data['ethical_reflection'].append(ethical_score)
        data['individual_voice_divergence'].append(sentiment_complexity)
        data['existential_awareness'].append(existential_score)
        data['emergent_agency_index'].append(eai)

    return pd.DataFrame(data)

