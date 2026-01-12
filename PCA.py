import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
data = {
    'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
    'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
    'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
    'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Male, 0 = Female
}

df = pd.DataFrame(data)
print(df)

# Features and target
X = df.drop('Gender', axis=1)
y = df['Gender']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Female', 'Male'],
            yticklabels=['Female', 'Male'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualization Before and After PCA
y_numeric = pd.factorize(y)[0]

plt.figure(figsize=(12, 5))

# Before PCA
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Height (Standardized)')
plt.ylabel('Weight (Standardized)')
plt.title('Before PCA')
plt.colorbar(label='Gender')

# After PCA
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('After PCA')
plt.colorbar(label='Gender')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

X, y_true = make_blobs(n_samples=300, centers=4,
 cluster_std=0.50, random_state=0)

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r']
print(colors)
for k, col in zip(unique_labels, colors):
 if k == -1:

  col = 'k'
 class_member_mask = (labels == k)
 xy = X[class_member_mask & core_samples_mask]
 plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
 markeredgecolor='k',
 markersize=6)
 xy = X[class_member_mask & ~core_samples_mask]
 plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
 markeredgecolor='k',
 markersize=6)
plt.title('number of clusters: %d' % n_clusters_)
plt.show()

