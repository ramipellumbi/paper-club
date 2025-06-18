import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

from data import make_moons
from spectral_clustering import spectral_clustering_k2


def compare_with_sklearn(X, y_true):
    """
    Compare manual k-NN implementation with sklearn's k-NN SpectralClustering.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Manual implementation - kNN
    clusters_manual_knn, _, _ = spectral_clustering_k2(X, n_neighbors=10)
    accuracy_manual_knn = max(np.mean(clusters_manual_knn == y_true), np.mean(clusters_manual_knn == (1 - y_true)))

    # sklearn - kNN affinity
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

    plt.suptitle("Manual vs sklearn k-NN SpectralClustering Comparison", fontsize=16)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Generate circles dataset
    X, y_true = make_moons(500)

    fig = compare_with_sklearn(X, y_true)
    plt.show()
