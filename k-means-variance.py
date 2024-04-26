import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        centroids_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[centroids_idx]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

# Generate random data for demonstration
np.random.seed(42)
X = np.random.rand(100, 2)

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit(X)

# Calculate variance for each cluster
variances = []
for k in range(kmeans.n_clusters):
    cluster_points = X[labels == k]
    cluster_variance = np.var(cluster_points, axis=0).mean()  # Compute variance for each feature and then take the mean
    variances.append(cluster_variance)

# Normalize variance values to percentages
max_variance = max(variances)
variances_percent = [(variance / max_variance) * 100 for variance in variances]

# Print variance for each cluster as percentages
for i, variance_percent in enumerate(variances_percent):
    print(f"Variance of Cluster {i+1}: {variance_percent:.2f}%")

# Plot the clustered data after K-means clustering
plt.figure(figsize=(8, 6))
for k in range(kmeans.n_clusters):
    cluster_points = X[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k+1}')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='black', marker='x', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
