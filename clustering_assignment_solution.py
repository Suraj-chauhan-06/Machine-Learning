# Clustering Assignment - Java + DSA | Pwskills

## Part 1: Theoretical Questions

# 1. What is unsupervised learning in the context of machine learning?
"""
Unsupervised learning involves training models on data without predefined labels. The model tries to identify patterns, such as grouping similar items or reducing dimensionality. Clustering is a common type of unsupervised learning.
"""

# 2. How does K-Means clustering algorithm work?
"""
K-Means clustering partitions the data into k clusters by minimizing the variance within each cluster:
1. Randomly initialize k centroids.
2. Assign each data point to the nearest centroid.
3. Recalculate centroids based on the mean of assigned points.
4. Repeat steps 2–3 until convergence.
"""

# 3. Explain the concept of a dendrogram in hierarchical clustering.
"""
A dendrogram is a tree-like diagram that illustrates the arrangement of clusters formed by hierarchical clustering. It shows how clusters are merged or split at each step.
"""

# 4. What is the main difference between K-Means and Hierarchical Clustering?
"""
K-Means is a partitional algorithm requiring the number of clusters in advance, while hierarchical clustering builds a tree of clusters without needing to predefine the number of clusters.
"""

# 5. What are the advantages of DBSCAN over K-Means?
"""
- Detects clusters of arbitrary shape
- Identifies outliers (noise points)
- Does not require the number of clusters in advance
"""

# 6. When would you use Silhouette Score in clustering?
"""
Silhouette Score measures how similar a point is to its own cluster compared to other clusters. It is used to evaluate the quality and separation of clusters.
"""

# 7. What are the limitations of Hierarchical Clustering?
"""
- High computational cost for large datasets
- Sensitive to noise and outliers
- No flexibility to change the number of clusters once built
"""

# 8. Why is feature scaling important in clustering algorithms like K-Means?
"""
K-Means uses Euclidean distance, which is sensitive to scale. Feature scaling ensures all features contribute equally to distance calculations.
"""

# 9. How does DBSCAN identify noise points?
"""
Points that do not belong to any dense region (i.e., fewer than min_samples within eps distance) are labeled as noise by DBSCAN.
"""

# 10. Define inertia in the context of K-Means.
"""
Inertia is the sum of squared distances between each point and its assigned cluster centroid. Lower inertia indicates more compact clusters.
"""

# 11. What is the elbow method in K-Means clustering?
"""
The elbow method involves plotting inertia against the number of clusters and identifying the point (elbow) where adding more clusters doesn't significantly reduce inertia.
"""

# 12. Describe the concept of "density" in DBSCAN.
"""
Density refers to the number of points within a specified radius (eps). DBSCAN uses this concept to group points and form clusters.
"""

# 13. Can hierarchical clustering be used on categorical data?
"""
Yes, with appropriate distance metrics (e.g., Hamming, Jaccard), hierarchical clustering can be applied to categorical data.
"""

# 14. What does a negative Silhouette Score indicate?
"""
A negative Silhouette Score indicates that a sample may have been assigned to the wrong cluster, as it is closer to points in another cluster.
"""

# 15. Explain the term "linkage criteria" in hierarchical clustering.
"""
Linkage criteria determine how distances between clusters are calculated. Common types include single, complete, average, and ward linkage.
"""

# 16. Why might K-Means clustering perform poorly on data with varying cluster sizes or densities?
"""
K-Means assumes clusters are spherical and equal in size. It struggles with clusters of different sizes, shapes, or densities.
"""

# 17. What are the core parameters in DBSCAN, and how do they influence clustering?
"""
- `eps`: Maximum distance for two points to be considered neighbors
- `min_samples`: Minimum number of neighbors to form a dense region
These parameters control cluster formation and noise detection.
"""

# 18. How does K-Means++ improve upon standard K-Means initialization?
"""
K-Means++ chooses initial centroids to be far apart, reducing the chance of poor clustering and improving convergence speed.
"""

# 19. What is agglomerative clustering?
"""
Agglomerative clustering is a bottom-up hierarchical method where each data point starts as a single cluster and merges iteratively based on distance.
"""

# 20. What makes Silhouette Score a better metric than just inertia for model evaluation?
"""
Silhouette Score accounts for both intra-cluster cohesion and inter-cluster separation, providing a more comprehensive assessment than inertia alone.
"""

# Clustering Assignment - Java + DSA | Pwskills

## Part 1: Theoretical Questions

# [Already Answered Above]

## Part 2: Practical Questions

### 1. Generate synthetic data with 4 centers using make_blobs and apply KMeans
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("KMeans on make_blobs with 4 centers")
plt.show()
```

### 2. Load the Iris dataset and use Agglomerative Clustering
```python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
X = iris.data
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(X)
print("First 10 predicted labels:", labels[:10])
```

### 3. Generate synthetic data using make_moons and apply DBSCAN
```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05)
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title("DBSCAN on make_moons")
plt.show()
```

### 4. Load the Wine dataset and apply KMeans after standardizing the features
```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

import numpy as np
unique, counts = np.unique(labels, return_counts=True)
print("Cluster sizes:", dict(zip(unique, counts)))
```

### 5. Use make_circles and cluster using DBSCAN
```python
from sklearn.datasets import make_circles

X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05)
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral')
plt.title("DBSCAN on make_circles")
plt.show()
```

### 6. Load Breast Cancer dataset, use MinMaxScaler and KMeans
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X = MinMaxScaler().fit_transform(cancer.data)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print("Cluster centroids:\n", kmeans.cluster_centers_)
```

### 7. Varying std devs with make_blobs and DBSCAN
```python
X, y = make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 1.5, 0.3], random_state=42)
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.title("DBSCAN on varying std devs")
plt.show()
```

### 8. Digits dataset + PCA + KMeans
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = PCA(n_components=2).fit_transform(digits.data)
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.title("KMeans on Digits with PCA")
plt.show()
```

### 9. Evaluate silhouette scores for k = 2 to 5
```python
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    scores.append(score)

plt.bar(range(2, 6), scores)
plt.title("Silhouette Scores for k=2 to 5")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()
```

### 10. Iris dataset + dendrogram (average linkage)
```python
import scipy.cluster.hierarchy as sch

X = load_iris().data
linkage_matrix = sch.linkage(X, method='average')

plt.figure(figsize=(10, 5))
sch.dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram (Average Linkage)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
```

### 11. Generate synthetic data with overlapping clusters using make_blobs and visualize KMeans with decision boundaries
```python
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='Pastel2', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1')
plt.title("KMeans with decision boundaries")
plt.show()
```

### 12. Load the Digits dataset and apply DBSCAN after reducing dimensions with t-SNE
```python
from sklearn.manifold import TSNE

X = TSNE(n_components=2, random_state=42).fit_transform(load_digits().data)
db = DBSCAN(eps=3, min_samples=5)
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.title("DBSCAN on Digits (t-SNE reduced)")
plt.show()
```

### 13. Generate data with make_blobs and apply Agglomerative Clustering with complete linkage
```python
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
agglo = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels = agglo.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20')
plt.title("Agglomerative Clustering (complete linkage)")
plt.show()
```

### 14. Load Breast Cancer dataset and compare inertia values for K = 2 to 6
```python
X = StandardScaler().fit_transform(load_breast_cancer().data)
inertias = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(2, 7), inertias, marker='o')
plt.title("KMeans Inertia for K=2 to 6")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```

### 15. Generate concentric circles and cluster with Agglomerative Clustering using single linkage
```python
X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05)
agglo = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = agglo.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Accent')
plt.title("Agglomerative Clustering on Circles (single linkage)")
plt.show()
```

### 16. Use the Wine dataset, apply DBSCAN after scaling, and count the number of clusters
```python
X = StandardScaler().fit_transform(load_wine().data)
db = DBSCAN(eps=1.2, min_samples=5)
labels = db.fit_predict(X)
unique_labels = set(labels)
print("Number of clusters (excluding noise):", len(unique_labels) - (1 if -1 in labels else 0))
```

### 17. Generate blobs and apply KMeans, then plot cluster centers
```python
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X')
plt.title("KMeans Clustering with Cluster Centers")
plt.show()
```

### 18. Load Iris dataset, cluster with DBSCAN, and print noise samples
```python
X = load_iris().data
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
print("Number of noise samples:", list(labels).count(-1))
```

### 19. Generate non-linearly separable data using make_moons and apply KMeans
```python
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='cool')
plt.title("KMeans on make_moons")
plt.show()
```

### 20. Load Digits dataset, apply PCA (3D), then KMeans and plot 3D scatter
```python
from mpl_toolkits.mplot3d import Axes3D

X = PCA(n_components=3).fit_transform(load_digits().data)
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='tab10')
ax.set_title("KMeans Clustering in 3D (Digits PCA)")
plt.show()
```

# Add more as per assignment questions — this template sets you up to complete everything with visual clarity.
