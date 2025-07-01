from textblob import TextBlob
import pandas as pd
import os

# Update the input file path to the latest available file
input_file = 'data/raw/relationships_posts_20250418_174400.jsonl'  # Use the most recent file
output_file = 'data/processed/reddit_with_sentiment.csv'

# Function to extract sentiment polarity and subjectivity
def extract_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Sentiment polarity (-1 to 1)
    subjectivity = blob.sentiment.subjectivity  # Subjectivity (0 to 1)
    return polarity, subjectivity

# Load your scraped Reddit posts data from the updated file
try:
    df = pd.read_json(input_file, lines=True)
    print(f"Successfully loaded data from {input_file}")
except ValueError as e:
    print(f"Error loading JSON file: {e}")
    exit()

# Apply sentiment extraction to each post
df[['sentiment_polarity', 'sentiment_subjectivity']] = df['selftext'].apply(lambda x: pd.Series(extract_sentiment(x)))

# Save the data with extracted features
df.to_csv(output_file, index=False)

print(f"Sentiment features extracted and saved to '{output_file}'")
