"""
============================================================
Epicon
============================================================

A lightweight, from-scratch machine learning library built on
NumPy with optional Numba acceleration. Provides a unified API
for neural networks and traditional ML models, designed to be
a minimal yet capable alternative to PyTorch/TensorFlow for
everyday ML tasks.

Key Design Principles:
    - Simple, consistent API (fit/predict) across all models
    - Minimal dependencies (NumPy required, Numba optional)
    - Educational transparency — readable, documented source
    - Fast execution via vectorized NumPy and optional Numba JIT

Quick Start:
    >>> import epicon
    >>> from epicon.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = epicon.LogisticRegression()
    >>> model.fit(X, y)
    >>> model.predict(X[:5])
"""

# ---------------------------------------------------------------------------
# Neural Network Components
# ---------------------------------------------------------------------------
from epicon.activations import Activation, LeakyReLU, ReLU, Sigmoid, Softmax, Tanh

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from epicon.datasets import load_iris, load_mnist, make_classification, make_regression
from epicon.ensemble import RandomForestClassifier, RandomForestRegressor
from epicon.layers import Conv1D, Dense, Dropout, Layer

# ---------------------------------------------------------------------------
# Traditional ML Models
# ---------------------------------------------------------------------------
from epicon.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from epicon.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from epicon.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from epicon.models import Model, Sequential
from epicon.naive_bayes import GaussianNB
from epicon.neighbors import KNeighborsClassifier, KNeighborsRegressor
from epicon.optimizers import Adam, GradientDescent, Momentum, Optimizer

# ---------------------------------------------------------------------------
# Preprocessing & Utilities
# ---------------------------------------------------------------------------
from epicon.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    train_test_split,
)
from epicon.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.3.0"
