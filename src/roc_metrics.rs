extern crate ndarray;
extern crate rayon;

use rayon::prelude::*;
use std::cmp::Ordering;

// Struct to represent a prediction element with score and target.
#[derive(Debug, Clone)]
pub struct PredictElem {
    score: f32,
    target: i32,
}

// Struct to hold the result of ROC calculation.
#[derive(Debug)]
pub struct RocData {
    pub tps: Vec<i32>,
    pub fps: Vec<i32>,
}

// Class-like structure for calculating ROC-based metrics.
pub struct RocMetrics {
    full_data: Vec<PredictElem>,
}

impl RocMetrics {
    // Constructor to initialize the RocMetrics object with scores and targets as slices.
    pub fn new(scores: &[f32], targets: &[i32]) -> RocMetrics {
        assert!(
            scores.len() > 1,
            "Scores array must be of length greater than 1."
        );
        assert!(
            targets.len() > 1,
            "Targets array must be of length greater than 1."
        );
        assert!(
            scores.len() == targets.len(),
            "Scores and targets must be of the same length."
        );

        let full_data: Vec<PredictElem> = scores
            .iter()
            .zip(targets.iter())
            .filter(|(_, &t)| t >= 0)
            .map(|(&s, &t)| PredictElem { score: s, target: t })
            .collect();

        println!("== roc_metrics: number of valid eval samples: {}", full_data.len());

        RocMetrics { full_data }
    }

    // Compute the ROC curve and return the raw vectors: tps and fps.
    pub fn binary_roc(&self) -> RocData {
        let mut tps: Vec<i32> = vec![0];
        let mut fps: Vec<i32> = vec![0];

        let mut prev_score = self.full_data[0].score;
        let mut accum = 0;
        let mut thresh_idx = 0;

        for elem in &self.full_data {
            let cur_score = elem.score;
            if cur_score != prev_score {
                tps.push(accum);
                fps.push(thresh_idx - accum);
            }
            prev_score = cur_score;
            accum += elem.target;
            thresh_idx += 1;
        }

        tps.push(accum);
        fps.push(thresh_idx - accum);

        RocData { tps, fps }
    }

    // Compute the Area Under the ROC Curve (AUC).
    pub fn compute_roc_auc(&mut self) -> f32 {
        // Parallel sort based on scores in descending order.
        self.full_data.par_sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let RocData { tps, fps } = self.binary_roc();
        let tp_count = tps.last().copied().unwrap_or(0) as f32;
        let fp_count = fps.last().copied().unwrap_or(0) as f32;

        let tpr: Vec<f32> = tps.iter().map(|&tp| tp as f32 / tp_count).collect();
        let fpr: Vec<f32> = fps.iter().map(|&fp| fp as f32 / fp_count).collect();

        trapz(&tpr, &fpr) as f32
    }
}

// Trapezoidal integration to compute the area under the curve.
fn trapz(y: &[f32], x: &[f32]) -> f64 {
    let mut ret = 0.0;
    let mut x_prev = x[0];
    let mut y_prev = y[0];

    for i in 1..y.len() {
        if x_prev == 1.0 {
            break;
        }
        if x[i] != x_prev {
            ret += 0.5 * (x[i] - x_prev) as f64 * (y_prev + y[i]) as f64;
        }
        x_prev = x[i];
        y_prev = y[i];
    }
    ret
}