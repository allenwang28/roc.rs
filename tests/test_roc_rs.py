import numpy as np
from roc_rs import PyRocMetrics

def test_roc_auc():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    targets = np.array([1, 1, 0, 1, 0], dtype=np.int32)
    roc_metrics = PyRocMetrics(scores, targets)
    auc = roc_metrics.compute_roc_auc()
    assert abs(auc - 0.75) < 1e-6
