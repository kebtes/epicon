"""
MNIST digit classification using Sequential + Adam.

Demonstrates the new Keras-like Sequential API with string-based
activation specification and the Adam optimizer.
"""

import numpy as np

from epicon.datasets import load_mnist
from epicon.layers import Dense
from epicon.losses import CategoricalCrossEntropy
from epicon.metrics import accuracy_score
from epicon.models import Sequential
from epicon.optimizers import Adam
from epicon.preprocessing import train_test_split

np.random.seed(42)

X, y = load_mnist(return_X_y=True)
X = X[:1000] / 255.0
y = y[:1000]

from epicon.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y).reshape(-1, 1)

ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2, random_state=42)

model = Sequential(
    [
        Dense(X_train.shape[1], 64, activation="relu"),
        Dense(64, 32, activation="relu"),
        Dense(32, 10, activation="softmax"),
    ]
)

model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

y_pred = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(f"\nTest accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
