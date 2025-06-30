import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)

# Does train_model return the expected estimator type?
def test_train_model_returns_random_forest():
    """
    train_model() should return a fitted sklearn.ensemble.RandomForestClassifier.
    """
    # Small synthetic dataset
    X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(
        model, RandomForestClassifier
    ), "train_model() did not return a RandomForestClassifier instance."


# Are precision, recall, and F1 computed correctly?

def test_compute_model_metrics_exact_values():
    """
    Given a simple ground-truth / prediction pair, compute_model_metrics()
    should return known, exact precision, recall, and F1.
    """
    from ml.model import compute_model_metrics  # local import to avoid circularity

    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    # Manually derived expected values
    expected_precision = 1.0          # TP = 2, FP = 0
    expected_recall = 2 / 3           # TP = 2, FN = 1
    expected_f1 = 0.8                 # 2 * P * R / (P + R)

    p, r, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(p, expected_precision), "Precision mismatch."
    assert np.isclose(r, expected_recall), "Recall mismatch."
    assert np.isclose(f1, expected_f1), "F1-score mismatch."


# Does inference() return a NumPy array of the correct length?

def test_inference_output_shape_and_type():
    """
    inference() should output a NumPy array whose length matches
    the number of input rows.
    """
    X = np.array([[0, 1], [1, 0], [0, 0]])
    y = np.array([0, 1, 0])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray), "inference() did not return a NumPy array."
    assert preds.shape[0] == X.shape[0], "Prediction length does not match input rows."
