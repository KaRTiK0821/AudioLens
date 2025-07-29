# -----------------------------------------------------------
# Spotify Genre Grouping Project
# Major Project for Internship Completion
# -----------------------------------------------------------

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------
# 2. Load Dataset
# -----------------------------------------------------------
data_path = "spotify dataset.csv"  # Change this if needed
df = pd.read_csv(data_path)

print("\n--- Basic Dataset Info ---")
print(df.shape)
print(df.head())

# -----------------------------------------------------------
# 3. Data Preprocessing
# -----------------------------------------------------------

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Focus on numerical features for clustering
# Common audio features in Spotify data
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Remove rows with missing numerical data
df = df.dropna(subset=numeric_cols)

# Normalize numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

print("\nData after scaling:", scaled_data.shape)

# -----------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# -----------------------------------------------------------

# Distribution plots for key features
plt.figure(figsize=(12, 6))
sns.histplot(df['danceability'], bins=30, color='green', kde=True)
plt.title("Distribution of Danceability")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Audio Features")
plt.show()

# -----------------------------------------------------------
# 5. K-Means Clustering
# -----------------------------------------------------------

# Determine optimal clusters using Elbow Method
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
plt.show()

# Let's pick k=5 (just as an example, adjust based on elbow plot)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

df['cluster'] = cluster_labels

# Silhouette Score for evaluation
sil_score = silhouette_score(scaled_data, cluster_labels)
print(f"\nSilhouette Score for {optimal_k} clusters: {sil_score:.3f}")

# -----------------------------------------------------------
# 6. PCA for 2D Visualization
# -----------------------------------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette="Set2", s=60)
plt.title("PCA Visualization of Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# -----------------------------------------------------------
# 7. Cluster Summary
# -----------------------------------------------------------
print("\n--- Cluster Summary ---")
cluster_summary = df.groupby('cluster')[numeric_cols].mean()
print(cluster_summary)

# -----------------------------------------------------------
# 8. Simple Recommendation Function
# -----------------------------------------------------------

def recommend_similar_songs(song_name, n_recommendations=5):
    if song_name not in df['track_name'].values:
        return f"Song '{song_name}' not found in dataset."
    
    # Get cluster of the given song
    song_cluster = df[df['track_name'] == song_name]['cluster'].iloc[0]
    
    # Get songs from the same cluster
    similar_songs = df[df['cluster'] == song_cluster]
    
    # Exclude the input song itself
    similar_songs = similar_songs[similar_songs['track_name'] != song_name]
    
    # Return top N similar songs
    return similar_songs[['track_name', 'track_artist', 'playlist_genre']].head(n_recommendations)

# Example usage
print("\nRecommendations for a sample song:")
print(recommend_similar_songs("Shape of You"))
