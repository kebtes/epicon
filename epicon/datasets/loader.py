"""
---------------------------------------------------------
DATASET LOADING UTILITIES
---------------------------------------------------------

Convenience functions for loading common benchmark datasets.
"""

from pathlib import Path

import numpy as np


def load_iris(return_X_y=False):
    """
    Load the Iris dataset.

    The Iris dataset contains 150 samples of iris flowers with
    4 features (sepal length, sepal width, petal length, petal width)
    and 3 target classes (setosa, versicolor, virginica).

    Parameters:
        return_X_y (bool): If True, returns (data, target) arrays
                           instead of a Bunch object. Defaults to False.

    Returns:
        Bunch or tuple: If return_X_y is False, returns a Bunch object
                       with attributes: data, target, target_names,
                       feature_names. If True, returns (data, target).

    Examples:
        >>> from epicon.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> X.shape
        (150, 4)
        >>> len(np.unique(y))
        3
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading Iris dataset. Install with: pip install pandas") from None

    # Look for the iris.csv in the resources directory
    base_path = Path(__file__).resolve().parent.parent.parent
    csv_path = base_path / "resources" / "iris" / "iris.csv"

    if not csv_path.exists():
        # Fallback: hardcoded iris dataset
        return _fallback_iris(return_X_y)

    df = pd.read_csv(csv_path)

    # Assume last column is target, rest are features
    feature_names = list(df.columns[:-1])
    target_names = np.unique(df.iloc[:, -1])

    X = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values

    # Map string labels to integers
    label_map = {name: i for i, name in enumerate(target_names)}
    y = np.array([label_map[name] for name in y_raw])

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, target_names=target_names, feature_names=feature_names, DESCR="Iris plants dataset")


def _fallback_iris(return_X_y=False):
    """
    Hardcoded Iris dataset when the CSV file is not found.
    """
    data = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.0, 1.4, 0.1],
            [4.3, 3.0, 1.1, 0.1],
            [5.8, 4.0, 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.5, 0.4],
            [4.6, 3.6, 1.0, 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5.0, 3.0, 1.6, 0.2],
            [5.0, 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.0, 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [4.4, 3.0, 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5.0, 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3.0, 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5.0, 3.3, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1.0],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
            [5.0, 2.0, 3.5, 1.0],
            [5.9, 3.0, 4.2, 1.5],
            [6.0, 2.2, 4.0, 1.0],
            [6.1, 2.9, 4.7, 1.4],
            [5.6, 2.9, 3.6, 1.3],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1],
            [5.9, 3.2, 4.8, 1.8],
            [6.1, 2.8, 4.0, 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [6.1, 2.8, 4.7, 1.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.6, 3.0, 4.4, 1.4],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3.0, 5.0, 1.7],
            [6.0, 2.9, 4.5, 1.5],
            [5.7, 2.6, 3.5, 1.0],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.4, 3.7, 1.0],
            [5.8, 2.7, 3.9, 1.2],
            [6.0, 2.7, 5.1, 1.6],
            [5.4, 3.0, 4.5, 1.5],
            [6.0, 3.4, 4.5, 1.6],
            [6.7, 3.1, 4.7, 1.5],
            [6.3, 2.3, 4.4, 1.3],
            [5.6, 3.0, 4.1, 1.3],
            [5.5, 2.5, 4.0, 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3.0, 4.6, 1.4],
            [5.8, 2.6, 4.0, 1.2],
            [5.0, 2.3, 3.3, 1.0],
            [5.6, 2.7, 4.2, 1.3],
            [5.7, 3.0, 4.2, 1.2],
            [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.9, 4.3, 1.3],
            [5.1, 2.5, 3.0, 1.1],
            [5.7, 2.8, 4.1, 1.3],
            [6.3, 3.3, 6.0, 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3.0, 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3.0, 5.8, 2.2],
            [7.6, 3.0, 6.6, 2.1],
            [4.9, 2.5, 4.5, 1.7],
            [7.3, 2.9, 6.3, 1.8],
            [6.7, 2.5, 5.8, 1.8],
            [7.2, 3.6, 6.1, 2.5],
            [6.5, 3.2, 5.1, 2.0],
            [6.4, 2.7, 5.3, 1.9],
            [6.8, 3.0, 5.5, 2.1],
            [5.7, 2.5, 5.0, 2.0],
            [5.8, 2.8, 5.1, 2.4],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 3.0, 5.5, 1.8],
            [7.7, 3.8, 6.7, 2.2],
            [7.7, 2.6, 6.9, 2.3],
            [6.0, 2.2, 5.0, 1.5],
            [6.9, 3.2, 5.7, 2.3],
            [5.6, 2.8, 4.9, 2.0],
            [7.7, 2.8, 6.7, 2.0],
            [6.3, 2.7, 4.9, 1.8],
            [6.7, 3.3, 5.7, 2.1],
            [7.2, 3.2, 6.0, 1.8],
            [6.2, 2.8, 4.8, 1.8],
            [6.1, 3.0, 4.9, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [7.2, 3.0, 5.8, 1.6],
            [7.4, 2.8, 6.1, 1.9],
            [7.9, 3.8, 6.4, 2.0],
            [6.4, 2.8, 5.6, 2.2],
            [6.3, 2.8, 5.1, 1.5],
            [6.1, 2.6, 5.6, 1.4],
            [7.7, 3.0, 6.1, 2.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 3.1, 5.5, 1.8],
            [6.0, 3.0, 4.8, 1.8],
            [6.9, 3.1, 5.4, 2.1],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3],
            [5.8, 2.7, 5.1, 1.9],
            [6.8, 3.2, 5.9, 2.3],
            [6.7, 3.3, 5.7, 2.5],
            [6.7, 3.0, 5.2, 2.3],
            [6.3, 2.5, 5.0, 1.9],
            [6.5, 3.0, 5.2, 2.0],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3.0, 5.1, 1.8],
        ]
    )

    target_names = np.array(["setosa", "versicolor", "virginica"])
    target = np.array([0] * 50 + [1] * 50 + [2] * 50)
    feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

    if return_X_y:
        return data, target

    return Bunch(
        data=data, target=target, target_names=target_names, feature_names=feature_names, DESCR="Iris plants dataset"
    )


def load_mnist(return_X_y=False):
    """
    Load the MNIST dataset from the resources directory.

    Parameters:
        return_X_y (bool): If True, returns (data, target) arrays
                           instead of a Bunch object. Defaults to False.

    Returns:
        Bunch or tuple: Dataset with attributes: data, target, DESCR.

    Examples:
        >>> from epicon.datasets import load_mnist
        >>> X, y = load_mnist(return_X_y=True)
        >>> X.shape[1]
        784
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading MNIST dataset.") from None

    base_path = Path(__file__).resolve().parent.parent.parent
    train_path = base_path / "resources" / "mnist" / "mnist_train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"MNIST data not found at {train_path}. "
            "Download from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
        )

    df = pd.read_csv(train_path)
    y = df.iloc[:, 0].values.astype(np.int64)
    X = df.iloc[:, 1:].values.astype(np.float64) / 255.0

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, DESCR="MNIST handwritten digits dataset")


class Bunch(dict):
    """
    A dictionary that exposes its keys as attributes.

    Examples:
        >>> b = Bunch(data=[1, 2], target=[0, 1])
        >>> b.data
        [1, 2]
        >>> b.target
        [0, 1]
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Bunch' object has no attribute '{key}'") from None

    def __setattr__(self, key, value):
        self[key] = value
