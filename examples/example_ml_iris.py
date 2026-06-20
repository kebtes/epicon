"""
Iris multi-class classification using DecisionTreeClassifier.

Demonstrates the simple fit/predict API with preprocessing,
train/test split, and accuracy scoring.
"""

import numpy as np
from epicon.datasets import load_iris
from epicon.preprocessing import train_test_split
from epicon.metrics import accuracy_score
from epicon.tree import DecisionTreeClassifier

np.random.seed(42)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"DecisionTree accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Feature importances: {model.feature_importances_}")
