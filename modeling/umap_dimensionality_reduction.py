import umap
import pandas as pd

# Load the processed data with sentiment features
df = pd.read_csv('/Users/am/python_code/project_folder/standalone_complex_profiler/data/processed/reddit_with_sentiment.csv')

# Select features (Can add more features if you have them)
features = df[['sentiment_polarity', 'sentiment_subjectivity']]  # Add any other features you want

# Initialize and fit UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
umap_embeddings = umap_model.fit_transform(features)

# Save the UMAP output
df['umap_x'] = umap_embeddings[:, 0]
df['umap_y'] = umap_embeddings[:, 1]

# Save the new data with UMAP coordinates
df.to_csv('/Users/am/python_code/project_folder/standalone_complex_profiler/data/processed/reddit_with_umap.csv', index=False)

print("UMAP dimensionality reduction complete and saved to 'data/processed/reddit_with_umap.csv'")
