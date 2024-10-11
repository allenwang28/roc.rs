import numpy as np
from sklearn.metrics import roc_auc_score
from roc_rs import PyRocMetrics
import pytest


def test_roc_auc():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    targets = np.array([1, 1, 0, 1, 0], dtype=np.int32)
    roc_metrics = PyRocMetrics(scores, targets)
    auc = roc_metrics.compute_roc_auc()
    expected_auc = 0.8333333730697632
    assert abs(auc - expected_auc) < 1e-6


def test_binary_roc():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    targets = np.array([1, 1, 0, 1, 0], dtype=np.int32)
    roc_metrics = PyRocMetrics(scores, targets)
    roc_data = roc_metrics.binary_roc()
    assert roc_data.tps == [0, 1, 2, 2, 3, 3]
    assert roc_data.fps == [0, 0, 0, 1, 1, 2]


def test_binary_roc_edge_cases():
    # All positive
    scores1 = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    targets1 = np.array([1, 1, 1], dtype=np.int32)
    roc_metrics1 = PyRocMetrics(scores1, targets1)
    roc_data1 = roc_metrics1.binary_roc()
    assert roc_data1.tps == [0, 1, 2, 3]
    assert roc_data1.fps == [0, 0, 0, 0]

    # All negative
    scores2 = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    targets2 = np.array([0, 0, 0], dtype=np.int32)
    roc_metrics2 = PyRocMetrics(scores2, targets2)
    roc_data2 = roc_metrics2.binary_roc()
    assert roc_data2.tps == [0, 0, 0, 0]
    assert roc_data2.fps == [0, 1, 2, 3]


@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_compare_with_sklearn(size):
    # Generate random data
    np.random.seed(42)
    scores = np.random.random(size).astype(np.float32)
    targets = np.random.randint(0, 2, size).astype(np.int32)

    # Calculate AUC using roc_rs
    roc_metrics = PyRocMetrics(scores, targets)
    roc_rs_auc = roc_metrics.compute_roc_auc()

    # Calculate AUC using scikit-learn
    sklearn_auc = roc_auc_score(targets, scores)

    # Compare results
    assert np.isclose(
        roc_rs_auc, sklearn_auc, rtol=1e-5, atol=1e-8
    ), f"AUC mismatch: roc_rs={roc_rs_auc}, sklearn={sklearn_auc}"
