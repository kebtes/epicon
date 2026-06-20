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
from epicon.models import Model, Sequential
from epicon.layers import Dense, Dropout, Conv1D, Layer
from epicon.activations import ReLU, LeakyReLU, Sigmoid, Softmax, Tanh, Activation
from epicon.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from epicon.optimizers import GradientDescent, Momentum, Adam, Optimizer

# ---------------------------------------------------------------------------
# Traditional ML Models
# ---------------------------------------------------------------------------
from epicon.linear_model import LinearRegression
from epicon.linear_model import LogisticRegression
from epicon.linear_model import Ridge
from epicon.linear_model import Lasso
from epicon.neighbors import KNeighborsClassifier, KNeighborsRegressor
from epicon.naive_bayes import GaussianNB
from epicon.tree import DecisionTreeClassifier, DecisionTreeRegressor
from epicon.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------------------------
# Preprocessing & Utilities
# ---------------------------------------------------------------------------
from epicon.preprocessing import StandardScaler, MinMaxScaler
from epicon.preprocessing import LabelEncoder, OneHotEncoder
from epicon.preprocessing import train_test_split

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from epicon.datasets import load_iris, load_mnist
from epicon.datasets import make_classification, make_regression

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from epicon.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.3.0"
