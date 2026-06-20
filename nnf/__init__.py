"""
============================================================
NNF — Neural Network Framework
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
    >>> import nnf
    >>> from nnf.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = nnf.LogisticRegression()
    >>> model.fit(X, y)
    >>> model.predict(X[:5])
"""

# ---------------------------------------------------------------------------
# Neural Network Components
# ---------------------------------------------------------------------------
from nnf.models import Model, Sequential
from nnf.layers import Dense, Dropout, Conv1D, Layer
from nnf.activations import ReLU, LeakyReLU, Sigmoid, Softmax, Tanh, Activation
from nnf.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from nnf.optimizers import GradientDescent, Momentum, Adam, Optimizer

# ---------------------------------------------------------------------------
# Traditional ML Models
# ---------------------------------------------------------------------------
from nnf.linear_model import LinearRegression
from nnf.linear_model import LogisticRegression
from nnf.linear_model import Ridge
from nnf.linear_model import Lasso
from nnf.neighbors import KNeighborsClassifier, KNeighborsRegressor
from nnf.naive_bayes import GaussianNB
from nnf.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nnf.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------------------------
# Preprocessing & Utilities
# ---------------------------------------------------------------------------
from nnf.preprocessing import StandardScaler, MinMaxScaler
from nnf.preprocessing import LabelEncoder, OneHotEncoder
from nnf.preprocessing import train_test_split

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from nnf.datasets import load_iris, load_mnist
from nnf.datasets import make_classification, make_regression

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from nnf.metrics import (
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
