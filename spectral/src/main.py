import matplotlib.pyplot as plt
import numpy as np

from compare_sklearn import compare_with_sklearn
from compute_adjacency_matrix import compute_adjacency_matrix
from data import make_circles, make_moons
from spectral_clustering import spectral_clustering_k2


def main():
    """
    Main function demonstrating manual spectral clustering implementation.
    """
    # Test on both datasets
    datasets = {"Circles": make_circles(500), "Moons": make_moons(500)}

    for name, (X, y_true) in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {name}")
        print(f"{'=' * 50}")

        # Basic spectral clustering with visualization
        print("\nRunning spectral clustering...")
        # Use k-NN with k=10 for both datasets
        clusters, fiedler_vector, fiedler_value = spectral_clustering_k2(X, n_neighbors=10)

        # Calculate accuracy (accounting for label switching)
        accuracy1 = np.mean(clusters == y_true)
        accuracy2 = np.mean(clusters == (1 - y_true))
        accuracy = max(accuracy1, accuracy2)

        print(f"Fiedler value (λ₂): {fiedler_value:.6f}")
        print(f"Clustering accuracy: {accuracy:.3f}")

        # Visualize the complete process
        fig1 = visualize_spectral_clustering(X, y_true, n_neighbors=10)
        fig1.suptitle(f"Spectral Clustering Process - {name} Dataset", fontsize=16)
        plt.show()

        # Compare different parameters
        fig2 = compare_parameters(X, y_true)
        fig2.suptitle(f"Parameter Comparison - {name} Dataset", fontsize=16)
        plt.show()

        # Show eigenvalue gap
        A = compute_adjacency_matrix(X, 10)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigenvalues, _ = np.linalg.eigh(L)
        eigenvalues = np.sort(eigenvalues)[:20]

        plt.figure(figsize=(8, 5))
        plt.plot(eigenvalues, "bo-", markersize=8)
        plt.axvline(x=1, color="red", linestyle="--", alpha=0.5, label=f"Fiedler value (λ₂={eigenvalues[1]:.4f})")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.title(f"Eigenvalue Spectrum - {name} Dataset")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Compare with sklearn
        fig3 = compare_with_sklearn(X, y_true)
        fig3.suptitle(f"sklearn Comparison - {name} Dataset", fontsize=16)
        plt.show()


def plot_spectral_clustering_results(X, y_true, y_pred, title="Spectral Clustering Results"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", edgecolors="black", linewidth=0.5, alpha=0.8)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    plt.colorbar(scatter1, ax=ax1)

    # Predictions
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", edgecolors="black", linewidth=0.5, alpha=0.8)
    ax2.set_title("Spectral Clustering")
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    plt.colorbar(scatter2, ax=ax2)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_spectral_clustering(X, y_true, n_neighbors=5):
    """
    Comprehensive visualization of spectral clustering process.
    """
    # Perform clustering
    clusters, fiedler_vector, fiedler_value = spectral_clustering_k2(
        X, n_neighbors=n_neighbors, use_normalized_laplacian=True
    )

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Original data with true labels
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", edgecolors="black", linewidth=0.5, s=50)
    ax1.set_title("Ground Truth Labels", fontsize=14)
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")

    # 2. Adjacency matrix visualization
    ax2 = plt.subplot(2, 3, 2)
    A = compute_adjacency_matrix(X, n_neighbors)
    im = ax2.imshow(A, cmap="Blues", aspect="auto")
    ax2.set_title(f"Adjacency Matrix (k={n_neighbors})", fontsize=14)
    ax2.set_xlabel("Node index")
    ax2.set_ylabel("Node index")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # 3. Fiedler vector values
    ax3 = plt.subplot(2, 3, 3)
    sorted_indices = np.argsort(fiedler_vector)
    ax3.scatter(
        range(len(fiedler_vector)), fiedler_vector[sorted_indices], c=fiedler_vector[sorted_indices], cmap="RdBu", s=10
    )
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_title(f"Fiedler Vector (λ₂={fiedler_value:.4f})", fontsize=14)
    ax3.set_xlabel("Node index (sorted)")
    ax3.set_ylabel("Fiedler vector value")

    # 4. Spectral clustering result
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X[:, 0], X[:, 1], c=clusters, cmap="tab10", edgecolors="black", linewidth=0.5, s=50)
    ax4.set_title("Spectral Clustering Result", fontsize=14)
    ax4.set_xlabel("X1")
    ax4.set_ylabel("X2")

    # 5. Fiedler vector mapped to original space
    ax5 = plt.subplot(2, 3, 5)
    scatter5 = ax5.scatter(X[:, 0], X[:, 1], c=fiedler_vector, cmap="RdBu", edgecolors="black", linewidth=0.5, s=50)
    ax5.set_title("Fiedler Vector Values in Original Space", fontsize=14)
    ax5.set_xlabel("X1")
    ax5.set_ylabel("X2")
    plt.colorbar(scatter5, ax=ax5, fraction=0.046)

    # 6. First few eigenvalues
    ax6 = plt.subplot(2, 3, 6)
    _, _, _ = spectral_clustering_k2(X, n_neighbors=n_neighbors)
    A = compute_adjacency_matrix(X, n_neighbors)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigenvalues, _ = np.linalg.eigh(L)
    eigenvalues = np.sort(eigenvalues)[:10]  # First 10 eigenvalues

    ax6.bar(range(len(eigenvalues)), eigenvalues)
    ax6.set_title("First 10 Eigenvalues of Laplacian", fontsize=14)
    ax6.set_xlabel("Eigenvalue index")
    ax6.set_ylabel("Eigenvalue")
    ax6.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="Fiedler value")
    ax6.legend()

    plt.tight_layout()
    return fig


def compare_parameters(X, y_true):
    """
    Compare different parameter choices for spectral clustering.
    """
    k_values = [3, 5, 10, 15, 20]
    n_plots = len(k_values)

    fig, axes = plt.subplots(2, n_plots, figsize=(20, 8))

    for i, k in enumerate(k_values):
        # Unnormalized Laplacian
        clusters_unnorm, fiedler_unnorm, _ = spectral_clustering_k2(X, n_neighbors=k, use_normalized_laplacian=False)

        # Normalized Laplacian
        clusters_norm, fiedler_norm, _ = spectral_clustering_k2(X, n_neighbors=k, use_normalized_laplacian=True)

        # Plot unnormalized
        axes[0, i].scatter(X[:, 0], X[:, 1], c=clusters_unnorm, cmap="tab10", s=30, alpha=0.7)
        axes[0, i].set_title(f"Unnormalized (k={k})")

        # Plot normalized
        axes[1, i].scatter(X[:, 0], X[:, 1], c=clusters_norm, cmap="tab10", s=30, alpha=0.7)
        axes[1, i].set_title(f"Normalized (k={k})")

        # Remove ticks for cleaner look
        for ax in [axes[0, i], axes[1, i]]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Spectral Clustering: Normalized vs Unnormalized Laplacian", fontsize=16)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
