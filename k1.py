
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

plt.figure()
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Number of clusters
k = 3
clusters = {}

np.random.seed(23)

# Initialize clusters
for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    clusters[idx] = {
        'center': center,
        'points': []
    }

# Plot initial centers
plt.scatter(X[:, 0], X[:, 1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.show()


# Distance function
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# Assign points to nearest cluster
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]

        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)

        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)

    return clusters


# Update cluster centers
def update_clusters(clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            clusters[i]['center'] = points.mean(axis=0)

        clusters[i]['points'] = []

    return clusters


# Predict cluster labels
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred


# Run one iteration of K-means
clusters = assign_clusters(X, clusters)
clusters = update_clusters(clusters)
pred = pred_cluster(X, clusters)

# Plot final result
plt.scatter(X[:, 0], X[:, 1], c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.show()



