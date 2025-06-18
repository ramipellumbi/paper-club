import numpy as np


# pulled from old course code
def make_circles(
    n_samples: int = 100,
    *,
    shuffle: bool = True,
    noise: float | None = None,
    random_state: int | None = None,
    factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D circles dataset for clustering/classification tasks.

    Creates two concentric circles with different radii.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points to generate (split evenly between circles).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float or None, default=None
        Standard deviation of Gaussian noise added to the data.
        If None, no noise is added.
    random_state : int or None, default=None
        Random seed for reproducibility.
    factor : float, default=0.5
        Scale factor between inner and outer circle radii (0 < factor < 1).

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        The generated samples (2D coordinates).
    y : np.ndarray of shape (n_samples,)
        The integer labels (0 for outer circle, 1 for inner circle).

    Raises
    ------
    ValueError
        If factor is not in the range (0, 1) or n_samples < 2.

    Examples
    --------
    >>> X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)
    >>> X.shape
    (200, 2)
    """
    if factor <= 0 or factor >= 1:
        raise ValueError(f"factor must be in (0, 1), got {factor}")

    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_state)

    # Split samples between inner and outer circles
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate evenly spaced angles for each circle
    # Use different starting angles to avoid alignment
    outer_angles = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)

    # Add slight angular offset to inner circle for better visualization
    if shuffle:
        inner_angles += np.pi / n_samples_in

    # Generate points on circles
    outer_x = np.cos(outer_angles)
    outer_y = np.sin(outer_angles)
    inner_x = factor * np.cos(inner_angles)
    inner_y = factor * np.sin(inner_angles)

    # Combine into single array
    x = np.vstack([np.concatenate([outer_x, inner_x]), np.concatenate([outer_y, inner_y])]).T

    # Create labels: 0 for outer circle, 1 for inner circle
    y = np.concatenate(
        [
            np.ones(n_samples_in, dtype=np.int32),
            np.zeros(n_samples_out, dtype=np.int32),
        ]
    )

    # Add noise if specified
    if noise is not None and noise > 0:
        noise_matrix = rng.normal(scale=noise, size=x.shape)
        x += noise_matrix

    # Shuffle the data if requested
    if shuffle:
        indices = rng.permutation(n_samples)
        x = x[indices]
        y = y[indices]

    return x, y


# pulled from old course code
def make_moons(
    n_samples: int = 100, *, shuffle: bool = True, noise: float | None = None, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D moons dataset for clustering/classification tasks.

    Creates two interleaving half circles.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points to generate (split evenly between moons).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float or None, default=None
        Standard deviation of Gaussian noise added to the data.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        The generated samples.
    y : np.ndarray of shape (n_samples,)
        The integer labels (0 or 1).
    """
    rng = np.random.RandomState(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate points for upper moon
    upper_angles = np.linspace(0, np.pi, n_samples_out)
    upper_x = np.cos(upper_angles)
    upper_y = np.sin(upper_angles)

    # Generate points for lower moon (shifted)
    lower_angles = np.linspace(0, np.pi, n_samples_in)
    lower_x = 1 - np.cos(lower_angles)
    lower_y = -np.sin(lower_angles) + 0.5

    x = np.vstack([np.concatenate([upper_x, lower_x]), np.concatenate([upper_y, lower_y])]).T

    y = np.concatenate([np.zeros(n_samples_out, dtype=np.int32), np.ones(n_samples_in, dtype=np.int32)])

    if noise is not None and noise > 0:
        x += rng.normal(scale=noise, size=x.shape)

    if shuffle:
        indices = rng.permutation(n_samples)
        x = x[indices]
        y = y[indices]

    return x, y
