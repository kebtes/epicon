# Epicon — Agents Context

## Project Description

Epicon is a lightweight, open source machine learning library for Python focused on simplicity, education, and rapid experimentation. The project provides from scratch implementations of machine learning algorithms and neural network components while maintaining a clean, beginner friendly API and minimal dependencies.

The primary goal of Epicon is to make machine learning concepts more accessible by exposing the underlying implementation details rather than abstracting them away behind complex frameworks. The library is designed for learning, prototyping, experimentation, and small to medium scale machine learning projects.

Key principles:

- Simplicity over complexity
- Educational value over excessive abstraction
- Minimal dependencies and lightweight installation
- Clear, maintainable, and well documented code
- Consistent developer experience across all modules

When contributing to Epicon, prioritize readability, correctness, and API consistency. New features should align with the project's educational focus and avoid introducing unnecessary complexity or heavy external dependencies.

## Technical Overview

- **Version**: 0.3.0 (alpha)
- **Python**: >=3.10
- **Dependencies**: NumPy >=1.21 (required), Numba >=0.56 (optional JIT acceleration), tabulate >=0.8
- **Build**: setuptools via `pyproject.toml`
- **Linting**: ruff (line-length 120, target py310, rules E/F/I/N/W/UP/B)
- **Type checking**: mypy (lenient — ignores missing imports, untyped defs)
- **Testing**: pytest (169+ unit tests)

### Architecture

```
epicon/
├── activations/    — ReLU, LeakyReLU, Sigmoid, Softmax, Tanh
├── datasets/       — load_iris, load_mnist, make_classification, make_regression
├── ensemble/       — RandomForestClassifier, RandomForestRegressor
├── layers/         — Dense, Dropout, Conv1D (base: Layer)
├── linear_model/   — LinearRegression, Ridge, Lasso, LogisticRegression
├── losses/         — MSE, BinaryCrossEntropy, CategoricalCrossEntropy (base: Loss)
├── metrics/        — accuracy, precision, recall, f1, confusion_matrix, MSE, MAE, R²
├── models/         — Model (layer orchestration), Sequential (Keras-style wrapper)
├── naive_bayes/    — GaussianNB
├── neighbors/      — KNeighborsClassifier, KNeighborsRegressor
├── optimizers/     — GradientDescent, Momentum, Adam (base: Optimizer)
├── preprocessing/  — StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, train_test_split
├── tree/           — DecisionTreeClassifier, DecisionTreeRegressor
├── utils/          — ModelBuilder, LAYER_REGISTERY
├── _jit.py         — conditional Numba JIT decorator and accelerated functions
└── __init__.py     — public API surface, version
```

### API Convention

All models follow `fit(X, y)` / `predict(X)` / `score(X, y)`. Fitted attributes use a trailing underscore (`coef_`, `intercept_`, `classes_`, `mean_`, `std_`, etc.) consistent with scikit-learn convention.

## Python Conventions

### Imports
- **Absolute imports only** — no relative imports. Order: standard library, third-party, then `epicon.*`, each group separated by a blank line.
- NumPy is always `import numpy as np` (never `from numpy import ...`).

### Naming
| Category | Convention | Example |
|---|---|---|
| Classes | PascalCase | `LogisticRegression`, `DecisionTreeClassifier` |
| Functions/methods | snake_case | `fit`, `predict`, `predict_proba`, `train_test_split` |
| Variables | snake_case | `n_samples`, `learning_rate` |
| Private/internal | leading underscore | `_sigmoid`, `_grow_tree`, `_jit_if_available` |
| Constants | UPPER_SNAKE_CASE | `HAS_NUMBA`, `LAYER_REGISTERY` |
| Fitted attributes | trailing underscore | `coef_`, `intercept_`, `classes_` |

### Docstrings
NumPy-style:
```python
def fit(self, X, y):
    """
    Short description.

    Parameters:
        X (np.ndarray): Description.
        y (np.ndarray): Description.

    Returns:
        self: The fitted model.
    """
```

### Type Hints
Gradual typing with Python 3.10+ union syntax (`str | None`, not `Optional[str]`). Return type annotations are used but not strictly enforced. Use `list[Layer]`, `dict[str, Any]`, etc.

### Input Sanitization
Convert inputs at the top of `fit()`/`predict()`/`transform()`:
```python
X = np.asarray(X, dtype=np.float64)
```

### JIT Acceleration
Conditional Numba JIT via `_jit.py`:
```python
from epicon._jit import _jit_if_available

@_jit_if_available
def _hot_loop(...):
    ...
```
Supports both bare `@_jit_if_available` and argument `@_jit_if_available(nopython=True)` usage. Falls back to pure Python when Numba is absent.

### Error Handling
- `ValueError` with descriptive f-strings for invalid parameters.
- Standard Python exceptions only (no custom exception classes).
- `warnings.warn` for non-fatal issues.
- `raise ... from e` for chained exceptions.

### Testing
- Tests mirror package structure under `tests/`.
- `np.testing.assert_array_equal` / `np.testing.assert_array_almost_equal` for array comparisons.
- Plain `assert` for shapes, boolean conditions, and inequalities.
- Fixtures are `@pytest.fixture` functions, seeded with `np.random.seed(42)`.
- Test functions: `test_<functionality>_<scenario>()`.
- Class-based test groups: `class Test<Component>:` with `def test_<scenario>(self, fixture):`.

## Agent Behavior

- Prioritize readability, correctness, and API consistency over clever optimizations.
- Use NumPy vectorized operations over explicit Python loops.
- Follow existing code patterns (import style, docstring format, error handling).
- Add tests for new features — mirror existing test structure.
- New features should align with the project's educational focus.
- Avoid introducing heavy external dependencies.
- Run `ruff` linting before committing code.
