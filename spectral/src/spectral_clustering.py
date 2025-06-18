import numpy as np

from .compute_adjacency_matrix import compute_adjacency_matrix


def spectral_clustering_k2(X, n_neighbors=5, use_normalized_laplacian=True):
    """
    Manual implementation of spectral clustering for k=2 clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data
    n_neighbors : int
        Number of nearest neighbors for adjacency matrix
    use_normalized_laplacian : bool
        Whether to use normalized vs unnormalized Laplacian

    Returns
    -------
    clusters : array of shape (n_samples,)
        Binary cluster assignments (0 or 1)
    eigenvector : array of shape (n_samples,)
        The Fiedler vector used for clustering
    eigenvalue : float
        The Fiedler value (2nd smallest eigenvalue)
    """
    # Step 1: Construct adjacency matrix
    A = compute_adjacency_matrix(X, n_neighbors)

    # Step 2: Compute degree matrix
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # Step 3: Construct graph Laplacian
    if use_normalized_laplacian:
        # Handle potential zero degrees
        degrees_sqrt_inv = np.zeros_like(degrees)
        non_zero_mask = degrees > 0
        degrees_sqrt_inv[non_zero_mask] = 1.0 / np.sqrt(degrees[non_zero_mask])
        D_sqrt_inv = np.diag(degrees_sqrt_inv)

        # L_norm = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
        L = np.eye(len(D)) - D_sqrt_inv @ A @ D_sqrt_inv
    else:
        # L = D - A
        L = D - A

    # Step 4: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # eigh is better for symmetric matrices

    # Step 5: Sort by eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 6: Get Fiedler vector (2nd smallest eigenvalue's eigenvector)
    fiedler_value = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]

    # Step 7: Cluster based on Fiedler vector
    # For k=2, we can use sign-based clustering
    # Using median split is more robust than 0 threshold
    median_val = np.median(fiedler_vector)
    clusters = (fiedler_vector > median_val).astype(int)

    return clusters, fiedler_vector, fiedler_value
