import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

from data import make_moons
from main import spectral_clustering_k2

# Generate circles dataset
X, y_true = make_moons(500)

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Manual implementation - kNN
print("Running manual k-NN spectral clustering...")
clusters_manual_knn, _, _ = spectral_clustering_k2(X, n_neighbors=10)
accuracy_manual_knn = max(np.mean(clusters_manual_knn == y_true), np.mean(clusters_manual_knn == (1 - y_true)))

# sklearn - kNN affinity
print("Running sklearn k-NN spectral clustering...")
sc_knn = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=42)
clusters_sklearn_knn = sc_knn.fit_predict(X)
accuracy_sklearn_knn = max(np.mean(clusters_sklearn_knn == y_true), np.mean(clusters_sklearn_knn == (1 - y_true)))

# Plot results
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", edgecolors="black", linewidth=0.5, s=50)
axes[0].set_title("Ground Truth")

axes[1].scatter(X[:, 0], X[:, 1], c=clusters_manual_knn, cmap="tab10", edgecolors="black", linewidth=0.5, s=50)
axes[1].set_title(f"Manual k-NN (k=10)\nAccuracy: {accuracy_manual_knn:.3f}")

axes[2].scatter(X[:, 0], X[:, 1], c=clusters_sklearn_knn, cmap="tab10", edgecolors="black", linewidth=0.5, s=50)
axes[2].set_title(f"sklearn k-NN (k=10)\nAccuracy: {accuracy_sklearn_knn:.3f}")

for ax in axes:
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

plt.suptitle("Manual vs sklearn k-NN SpectralClustering on Circles Dataset", fontsize=16)
plt.tight_layout()

# Print accuracy summary
print("\nAccuracy Summary:")
print(f"Manual k-NN (k=10):  {accuracy_manual_knn:.3f}")
print(f"sklearn k-NN (k=10): {accuracy_sklearn_knn:.3f}")

plt.show()

# Timing comparison
print("\nTiming comparison (100 iterations):")
n_iter = 100

# Manual k-NN
start = time.time()
for _ in range(n_iter):
    spectral_clustering_k2(X, n_neighbors=10)
manual_time = (time.time() - start) / n_iter

# sklearn k-NN
start = time.time()
for _ in range(n_iter):
    sc_knn.fit_predict(X)
sklearn_time = (time.time() - start) / n_iter

print(f"Manual k-NN:  {manual_time * 1000:.2f} ms per iteration")
print(f"sklearn k-NN: {sklearn_time * 1000:.2f} ms per iteration")
print(f"sklearn is {manual_time / sklearn_time:.1f}x faster")
