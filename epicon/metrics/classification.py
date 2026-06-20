"""
---------------------------------------------------------
CLASSIFICATION METRICS
---------------------------------------------------------

Standard metrics for evaluating classification models.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy classification score.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Accuracy score (fraction of correct predictions).

    Examples:
        >>> from epicon.metrics import accuracy_score
        >>> accuracy_score([0, 1, 0, 1], [0, 1, 1, 1])
        0.75
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list or None): List of class labels. If None, inferred
                               from the sorted unique values of y_true.

    Returns:
        np.ndarray: Confusion matrix of shape (n_classes, n_classes).

    Examples:
        >>> from epicon.metrics import confusion_matrix
        >>> confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1])
        array([[1, 1],
               [0, 2]])
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        i = label_to_idx.get(t, -1)
        j = label_to_idx.get(p, -1)
        if i >= 0 and j >= 0:
            matrix[i, j] += 1

    return matrix


def precision_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Compute the precision score.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method ('binary', 'macro', 'micro',
                       'weighted'). Defaults to 'binary'.
        pos_label (int): The positive class label for binary classification.
                         Defaults to 1.

    Returns:
        float: Precision score.

    Examples:
        >>> from epicon.metrics import precision_score
        >>> precision_score([0, 1, 0, 1], [0, 1, 1, 1])
        0.666...
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == 'binary':
        if len(np.unique(y_true)) > 2 and pos_label not in np.unique(y_true):
            raise ValueError("Binary precision requires binary labels or a specified pos_label.")
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    labels = np.unique(y_true)
    scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

    if average == 'macro':
        return np.mean(scores)
    elif average == 'micro':
        tp = np.sum(y_pred == y_true)
        fp = np.sum(y_pred != y_true)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == label) for label in labels])
        return np.average(scores, weights=weights)
    else:
        raise ValueError(f"Unknown average method '{average}'")


def recall_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Compute the recall score.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method ('binary', 'macro', 'micro',
                       'weighted'). Defaults to 'binary'.
        pos_label (int): The positive class label for binary classification.
                         Defaults to 1.

    Returns:
        float: Recall score.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == 'binary':
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    labels = np.unique(y_true)
    scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    if average == 'macro':
        return np.mean(scores)
    elif average == 'micro':
        tp = np.sum(y_pred == y_true)
        fn = np.sum((y_pred != y_true) & (y_true != y_pred))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == label) for label in labels])
        return np.average(scores, weights=weights)
    else:
        raise ValueError(f"Unknown average method '{average}'")


def f1_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Compute the F1 score (harmonic mean of precision and recall).

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): Averaging method ('binary', 'macro', 'micro',
                       'weighted'). Defaults to 'binary'.
        pos_label (int): The positive class label for binary classification.
                         Defaults to 1.

    Returns:
        float: F1 score.
    """
    prec = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, average=average, pos_label=pos_label)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)
