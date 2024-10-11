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
