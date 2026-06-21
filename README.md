<p align="center">
  <img src="BANNER.png" alt="Epicon banner" width="100%">
</p>

[**EPICON**](https://epicon-web.vercel.app/), a **lightweight, from-scratch** machine learning library built on NumPy with optional Numba acceleration. Provides a unified API for neural networks
**and** traditional ML models.

Designed to be minimal yet capable — like **Flask for ML**.

## Quick Start

```python
import epicon
from epicon.datasets import load_iris
from epicon.preprocessing import train_test_split
from epicon.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a model
model = epicon.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## What's Included

### Neural Networks
- Layers: `Dense`, `Dropout`, `Conv1D`
- Activations: `ReLU`, `LeakyReLU`, `Sigmoid`, `Softmax`, `Tanh`
- Losses: `MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`
- Optimizers: `GradientDescent`, `Momentum`, `Adam`
- `Model` — layer-by-layer construction
- `Sequential` — Keras-style wrapper with string activations

### Traditional ML Models
- `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`
- `KNeighborsClassifier`, `KNeighborsRegressor`
- `GaussianNB`
- `DecisionTreeClassifier`, `DecisionTreeRegressor`
- `RandomForestClassifier`, `RandomForestRegressor`

### Utilities
- Preprocessing: `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`, `train_test_split`
- Datasets: `load_iris`, `load_mnist`, `make_classification`, `make_regression`
- Metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `mean_squared_error`, `mean_absolute_error`, `r2_score`

## Installation

```bash
# Minimal install (NumPy required)
pip install numpy

# Install Epicon from source
pip install -e .

# With Numba (optional, for faster tree/KNN)
pip install numba
```

## Design

- **Consistent API**: all models follow `fit(X, y)` / `predict(X)`.
- **Minimal dependencies**: only NumPy is required.
- **Optional acceleration**: Numba JIT for tree split search and KNN.
- **Educational**: readable, fully documented source code.
- **Tested**: 169+ unit tests with pytest.

## Examples

See the [examples/](examples/) directory:

- `example_ml_iris.py` — Decision tree on Iris dataset
- `example_ml_binary.py` — LogisticRegression with L2 penalty
- `example_ml_forest.py` — RandomForestRegressor on synthetic data
- `example_nn_sequential.py` — Sequential neural net with Adam on MNIST
