"""
Binary classification using LogisticRegression with L2 penalty.

Demonstrates fit/predict_proba API and regularization on a
synthetic binary classification task.
"""

import numpy as np
from epicon.datasets import make_classification
from epicon.preprocessing import train_test_split, StandardScaler
from epicon.metrics import accuracy_score, f1_score
from epicon.linear_model import LogisticRegression

np.random.seed(42)

X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                            n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.1, epochs=1000, C=0.5, penalty='l2')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 score: {f1_score(y_test, y_pred):.4f}")
print(f"Sample probabilities (first 5):\n{probs[:5].round(4)}")
