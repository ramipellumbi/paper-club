import numpy as np


def compute_adjacency_matrix(x, k):
    """
    [x]: Euclidean dataset
    [k]: amount of neighbors for each node

    returns adjacency matrix of X connecting each node
    to its k-nearest neighbors
    """
    n, _ = x.shape
    # initialize adjacency matrix
    a = np.zeros((n, n))

    # get KNN matrix
    d = _compute_distance_matrix(x)

    # For each point, find k nearest neighbors
    for i in range(n):
        # Get indices of k nearest neighbors (excluding self)
        neighbors = np.argsort(d[i, :])[1 : k + 1]
        a[i, neighbors] = 1

    a = np.maximum(a, a.T)

    return a


def _compute_distance_matrix(x):
    """
    [x]: Circles dataset with n nodes

    returns n*n matrix D where D(i,j) is
    the distance between node i and node j
    """
    n, _ = x.shape
    d = np.zeros((n, n))  # D[i,i] = 0 for all i

    for i in range(n):
        for j in range(i + 1, n):
            dist = _euclidean_distance(x[i], x[j])
            d[i][j] = dist
            d[j][i] = dist

    return d


def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the Euclidean distance between two vectors
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))
