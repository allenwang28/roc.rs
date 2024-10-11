mod roc_metrics;

use roc_metrics::RocMetrics;

fn main() {
    // Define sample data for testing.
    let scores_input = vec![0.1, 0.4, 0.35, 0.8];
    let tgts_input = vec![0, 0, 1, 1];

    // Create a new RocMetrics instance.
    let mut roc_obj = RocMetrics::new(&scores_input, &tgts_input);

    // Calculate AUC.
    let auc = roc_obj.compute_roc_auc();

    // Print out the AUC value.
    println!("Computed AUC: {:.2}", auc);

    // Compute and display the raw ROC values.
    let roc_data = roc_obj.binary_roc();
    println!("TPS: {:?}", roc_data.tps);
    println!("FPS: {:?}", roc_data.fps);
}