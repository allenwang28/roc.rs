import numpy as np
import time
from sklearn.metrics import roc_auc_score
from roc_rs import PyRocMetrics
import matplotlib.pyplot as plt
import pandas as pd

def generate_data(size):
    np.random.seed(42)
    scores = np.random.random(size).astype(np.float32)
    targets = np.random.randint(0, 2, size).astype(np.int32)
    return scores, targets

def benchmark_roc_rs(scores, targets):
    start_time = time.time()
    roc_metrics = PyRocMetrics(scores, targets)
    auc = roc_metrics.compute_roc_auc()
    end_time = time.time()
    return end_time - start_time, auc

def benchmark_sklearn(scores, targets):
    start_time = time.time()
    auc = roc_auc_score(targets, scores)
    end_time = time.time()
    return end_time - start_time, auc

sizes = [1000, 10000, 100000, 1000000, 10000000]
results = []

for size in sizes:
    print(f"Benchmarking with {size} samples...")
    scores, targets = generate_data(size)
    
    roc_rs_time, roc_rs_auc = benchmark_roc_rs(scores, targets)
    sklearn_time, sklearn_auc = benchmark_sklearn(scores, targets)
    
    results.append({
        'size': size,
        'roc_rs_time': roc_rs_time,
        'sklearn_time': sklearn_time,
        'roc_rs_auc': roc_rs_auc,
        'sklearn_auc': sklearn_auc,
    })
    
    print(f"ROC_RS Time: {roc_rs_time:.4f}s, AUC: {roc_rs_auc:.4f}")
    print(f"Sklearn Time: {sklearn_time:.4f}s, AUC: {sklearn_auc:.4f}")
    print()

# Convert results to DataFrame
df = pd.DataFrame(results)

# Plot time comparison
plt.figure(figsize=(10, 6))
plt.plot(df['size'], df['roc_rs_time'], marker='o', label='ROC_RS')
plt.plot(df['size'], df['sklearn_time'], marker='o', label='Sklearn')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('ROC AUC Computation Time: ROC_RS vs Sklearn')
plt.legend()
plt.grid(True)
plt.savefig('assets/benchmark_time.png')
plt.close()

# Plot AUC comparison
plt.figure(figsize=(10, 6))
plt.plot(df['size'], df['roc_rs_auc'], marker='o', label='ROC_RS')
plt.plot(df['size'], df['sklearn_auc'], marker='o', label='Sklearn')
plt.xscale('log')
plt.xlabel('Number of Samples')
plt.ylabel('AUC')
plt.title('ROC AUC Values: ROC_RS vs Sklearn')
plt.legend()
plt.grid(True)
plt.savefig('assets/benchmark_auc.png')
plt.close()

# Print results table
print(df.to_string(index=False))
