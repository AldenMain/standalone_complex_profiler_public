import pandas as pd

# Load the clustered data
df = pd.read_csv('data/processed/reddit_with_clusters.csv')

# Example: View cluster summaries (mean sentiment values per cluster)
cluster_summary = df.groupby('cluster')[['sentiment_polarity', 'sentiment_subjectivity']].mean()
print(cluster_summary)

# Example: View posts in Cluster 0 (replace with your actual clusters)
cluster_0_posts = df[df['cluster'] == 0][['title', 'selftext', 'sentiment_polarity', 'sentiment_subjectivity']]
print(cluster_0_posts.head())

# Define human-readable labels for clusters
cluster_labels = {
    0: 'Positive and Motivational Posts',
    1: 'Frustration and Anger',
    2: 'Neutral, Informational Posts',
    # Add more as necessary
}

# Add human-readable labels to the dataframe
df['cluster_label'] = df['cluster'].map(cluster_labels)

# Save the updated data
df.to_csv('data/processed/reddit_with_refined_labels.csv', index=False)

print("Cluster interpretation complete and saved with human-readable labels.")
