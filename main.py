# -----------------------------------------------------------
# AudioLens: Spotify Track Analysis & Clustering
# Updated Version - Saves plots and results in outputs folder
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# -----------------------------------------------------------
# 1. Setup & Load Dataset
# -----------------------------------------------------------
data_path = "spotify dataset.csv"  # Change if needed
df = pd.read_csv(data_path)

# Create output directory for saving results
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

print("\n--- Basic Dataset Info ---")
print(df.shape)
print(df.head())

# -----------------------------------------------------------
# 2. Data Preprocessing
# -----------------------------------------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Select numeric columns for clustering
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Drop rows with missing numeric data
df = df.dropna(subset=numeric_cols)

# Normalize numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

print("\nData after scaling:", scaled_data.shape)

# -----------------------------------------------------------
# 3. EDA and Visualizations (Save Plots)
# -----------------------------------------------------------

# Distribution of Danceability
plt.figure(figsize=(12, 6))
sns.histplot(df['danceability'], bins=30, color='green', kde=True)
plt.title("Distribution of Danceability")
plt.savefig(os.path.join(output_dir, "danceability_distribution.png"))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Audio Features")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# -----------------------------------------------------------
# 4. K-Means Clustering
# -----------------------------------------------------------

# Elbow Method
wcss = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig(os.path.join(output_dir, "elbow_method.png"))
plt.show()

# Select number of clusters (can adjust based on elbow plot)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
df['cluster'] = cluster_labels

# Silhouette Score
sil_score = silhouette_score(scaled_data, cluster_labels)
print(f"\nSilhouette Score for {optimal_k} clusters: {sil_score:.3f}")

# -----------------------------------------------------------
# 5. PCA for Visualization
# -----------------------------------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette="Set2", s=60)
plt.title("PCA Visualization of Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(os.path.join(output_dir, "pca_clusters.png"))
plt.show()

# -----------------------------------------------------------
# 6. Cluster Summary & Save as CSV
# -----------------------------------------------------------
print("\n--- Cluster Summary ---")
cluster_summary = df.groupby('cluster')[numeric_cols].mean()
print(cluster_summary)

# Save cluster summary to CSV
cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))

# -----------------------------------------------------------
# 7. Recommendation Function
# -----------------------------------------------------------
def recommend_similar_songs(song_name, n_recommendations=5):
    if song_name not in df['track_name'].values:
        return f"Song '{song_name}' not found in dataset."
    
    # Get cluster of given song
    song_cluster = df[df['track_name'] == song_name]['cluster'].iloc[0]
    
    # Get songs from same cluster
    similar_songs = df[df['cluster'] == song_cluster]
    
    # Exclude input song
    similar_songs = similar_songs[similar_songs['track_name'] != song_name]
    
    return similar_songs[['track_name', 'track_artist', 'playlist_genre']].head(n_recommendations)

# Example usage
print("\nRecommendations for a sample song:")
print(recommend_similar_songs("Shape of You"))

print(f"\nAll plots and cluster summary saved in folder: '{output_dir}'")
