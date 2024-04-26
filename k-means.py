import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Центроидуудыг санамсаргүй байдлаар сонгоно
        centroids_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[centroids_idx]

        for _ in range(self.max_iter):
            # Өгөгдлийн цэг бүрийг хамгийн ойрын төв хэсэгт онооно
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Центроидуудыг шинэчлэнэ
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Ойртож байгаа эсэхийг шалгана
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

# Санамсаргүй өгөгдөл үүсгэх
np.random.seed(42)
X = np.random.rand(100, 2)

# K-means алгоритм ажиллахын өмнөх диаграммыг үүсгэнэ
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data points')
plt.title('Dataset Before K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('kmeans_before.png')
plt.close()

# KMeans-ийг эхлүүлж, тохируулна
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit(X)

# К-means кластер болгосны дараа кластерлагдсан өгөгдлийг зурна
plt.figure(figsize=(8, 6))
for k in range(kmeans.n_clusters):
    cluster_points = X[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k+1}')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='black', marker='x', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('kmeans_after.png')
plt.close()

# Өгөгдлийн цэгийн тоог нэмэгдүүлэнэ
X = np.vstack([X, np.random.rand(50, 2) + 1.5])  # Өөр 50 санамсаргүй өгөгдлийн цэг нэмнэ

# Өгөгдлийн цэгүүдийг нэмэгдүүлсний дараа өгөгдлийн багцыг зурна
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data points')
plt.title('Dataset After Increasing Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('kmeans_after_data_increase.png')
plt.close()

# Дата цэгүүдийг нэмэгдүүлсэн KMeans-ийг дахин эхлүүлж, тохируулна
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit(X)

# Өгөгдлийн цэгүүдийг нэмэгдүүлсэн K-means кластерын дараа кластерлагдсан өгөгдлийг зур
plt.figure(figsize=(8, 6))
for k in range(kmeans.n_clusters):
    cluster_points = X[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k+1}')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='black', marker='x', label='Centroids')
plt.title('K-means Clustering After Increasing Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('kmeans_after_data_increase_result.png')
plt.close()

print("Plots saved as 'kmeans_before.png', 'kmeans_after.png', 'kmeans_after_data_increase.png', and 'kmeans_after_data_increase_result.png'.")