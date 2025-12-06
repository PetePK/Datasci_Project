"""
Quick test: BERTopic with DBSCAN clustering
"""
import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from sklearn.cluster import DBSCAN

print("Loading data...")
df = pd.read_parquet('../data/processed/papers.parquet')
embeddings = np.load('../data/embeddings/paper_embeddings.npy')

print(f"Loaded {len(df):,} papers with embeddings shape {embeddings.shape}")

# Test DBSCAN with BERTopic on small sample
print("\nTesting DBSCAN clustering on 1000 papers...")

sample_indices = np.random.choice(len(df), size=1000, replace=False)
sample_df = df.iloc[sample_indices].reset_index(drop=True)
sample_embeddings = embeddings[sample_indices]

# Configure BERTopic with DBSCAN
umap_model = UMAP(
    n_components=5,
    metric='cosine',
    random_state=42,
    n_neighbors=15
)

cluster_model = DBSCAN(
    eps=0.5,
    min_samples=15,  # Lower for small sample
    metric='euclidean'
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    verbose=True
)

print("\nFitting BERTopic with DBSCAN...")
topics, _ = topic_model.fit_transform(
    sample_df['abstract'].tolist(),
    embeddings=sample_embeddings
)

# Results
topic_info = topic_model.get_topic_info()
print(f"\nSUCCESS! Discovered {len(topic_info)} topics")
print(f"\nTopic distribution:")
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

# Check for outliers
outliers = (topics == -1).sum()
print(f"\nOutliers (Topic -1): {outliers} papers ({outliers/len(sample_df)*100:.1f}%)")

print("\nDBSCAN works perfectly with BERTopic!")
