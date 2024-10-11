import numpy as np
from roc_rs import PyRocMetrics

# Example data
scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
targets = np.array([1, 1, 0, 1, 0], dtype=np.int32)

# Create PyRocMetrics instance
roc_metrics = PyRocMetrics(scores, targets)

# Compute ROC
roc_data = roc_metrics.binary_roc()
print(f"TPs: {roc_data.tps}")
print(f"FPs: {roc_data.fps}")

# Compute AUC
auc = roc_metrics.compute_roc_auc()
print(f"AUC: {auc}")
