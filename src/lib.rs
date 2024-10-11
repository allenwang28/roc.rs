use numpy::PyArray1;
use pyo3::prelude::*;

mod roc_metrics;
use roc_metrics::RocMetrics;

#[pyclass]
struct PyRocData {
    #[pyo3(get)]
    tps: Vec<i32>,
    #[pyo3(get)]
    fps: Vec<i32>,
}

#[pymethods]
impl PyRocData {
    #[new]
    fn new(tps: Vec<i32>, fps: Vec<i32>) -> Self {
        PyRocData { tps, fps }
    }

    fn __repr__(&self) -> String {
        format!("RocData(tps={:?}, fps={:?})", self.tps, self.fps)
    }
}

#[pyclass]
struct PyRocMetrics {
    inner: RocMetrics,
}

#[pymethods]
impl PyRocMetrics {
    #[new]
    fn new(_py: Python<'_>, scores: &PyArray1<f32>, targets: &PyArray1<i32>) -> PyResult<Self> {
        let scores_slice = unsafe { scores.as_slice()? };
        let targets_slice = unsafe { targets.as_slice()? };
        let inner = RocMetrics::new(scores_slice, targets_slice);
        Ok(PyRocMetrics { inner })
    }

    fn binary_roc(&self) -> PyResult<PyRocData> {
        let data = self.inner.binary_roc();
        Ok(PyRocData::new(data.tps, data.fps))
    }

    fn compute_roc_auc(&mut self) -> f32 {
        self.inner.compute_roc_auc()
    }
}

#[pymodule]
fn roc_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRocMetrics>()?;
    m.add_class::<PyRocData>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roc_auc() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let targets = vec![1, 1, 0, 1, 0];
        let mut roc_metrics = RocMetrics::new(&scores, &targets);
        let auc = roc_metrics.compute_roc_auc();
        assert!((auc - 0.8333333730697632).abs() < 1e-6);
    }

    #[test]
    fn test_binary_roc() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let targets = vec![1, 1, 0, 1, 0];
        let roc_metrics = RocMetrics::new(&scores, &targets);
        let roc_data = roc_metrics.binary_roc();
        assert_eq!(roc_data.tps, vec![0, 1, 2, 2, 3, 3]);
        assert_eq!(roc_data.fps, vec![0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_binary_roc_edge_cases() {
        // All positive
        let scores1 = vec![0.9, 0.8, 0.7];
        let targets1 = vec![1, 1, 1];
        let roc_metrics1 = RocMetrics::new(&scores1, &targets1);
        let roc_data1 = roc_metrics1.binary_roc();
        assert_eq!(roc_data1.tps, vec![0, 1, 2, 3]);
        assert_eq!(roc_data1.fps, vec![0, 0, 0, 0]);

        // All negative
        let scores2 = vec![0.9, 0.8, 0.7];
        let targets2 = vec![0, 0, 0];
        let roc_metrics2 = RocMetrics::new(&scores2, &targets2);
        let roc_data2 = roc_metrics2.binary_roc();
        assert_eq!(roc_data2.tps, vec![0, 0, 0, 0]);
        assert_eq!(roc_data2.fps, vec![0, 1, 2, 3]);
    }
}