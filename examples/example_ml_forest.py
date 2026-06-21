"""
Regression using RandomForestRegressor.

Demonstrates ensemble training on synthetic data with
feature importance analysis and R² scoring.
"""

import numpy as np

from epicon.datasets import make_regression
from epicon.ensemble import RandomForestRegressor
from epicon.metrics import mean_squared_error, r2_score
from epicon.preprocessing import train_test_split

np.random.seed(42)

X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Feature importances: {model.feature_importances_}")
