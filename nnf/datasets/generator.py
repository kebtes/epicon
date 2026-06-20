"""
---------------------------------------------------------
SYNTHETIC DATASET GENERATORS
---------------------------------------------------------

Functions for generating synthetic datasets for testing
and experimentation.
"""

import numpy as np


def make_regression(n_samples=100, n_features=10, noise=0.1, random_state=None):
    """
    Generate a random regression dataset.

    Parameters:
        n_samples (int): Number of samples. Defaults to 100.
        n_features (int): Number of features. Defaults to 10.
        noise (float): Standard deviation of Gaussian noise added
                       to the output. Defaults to 0.1.
        random_state (int or None): Random seed. Defaults to None.

    Returns:
        tuple: (X, y) where X has shape (n_samples, n_features)
               and y has shape (n_samples,).

    Examples:
        >>> from nnf.datasets import make_regression
        >>> X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        >>> X.shape
        (50, 5)
        >>> y.shape
        (50,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + noise * np.random.randn(n_samples)

    return X, y


def make_classification(n_samples=100, n_features=10, n_classes=2, n_informative=5, random_state=None):
    """
    Generate a random classification dataset.

    Parameters:
        n_samples (int): Number of samples. Defaults to 100.
        n_features (int): Total number of features. Defaults to 10.
        n_classes (int): Number of classes. Defaults to 2.
        n_informative (int): Number of informative features. Defaults to 5.
        random_state (int or None): Random seed. Defaults to None.

    Returns:
        tuple: (X, y) where X has shape (n_samples, n_features)
               and y has shape (n_samples,).

    Examples:
        >>> from nnf.datasets import make_classification
        >>> X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        >>> X.shape
        (50, 4)
        >>> len(np.unique(y))
        2
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)

    # Generate informative features as linear combinations of latent factors
    n_informative = min(n_informative, n_features)
    latent = np.random.randn(n_samples, n_informative)
    coef = np.random.randn(n_informative, n_classes)

    # Compute class scores
    scores = latent @ coef
    y = np.argmax(scores, axis=1)

    # Replace some features with noise
    if n_informative < n_features:
        X[:, :n_informative] = latent[:, :n_informative] + 0.1 * np.random.randn(n_samples, n_informative)
        X[:, n_informative:] = np.random.randn(n_samples, n_features - n_informative)
    else:
        X = latent + 0.1 * np.random.randn(n_samples, n_features)

    return X, y
