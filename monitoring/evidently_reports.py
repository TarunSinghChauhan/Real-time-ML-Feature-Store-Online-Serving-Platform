import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
REPORT_DIR = os.path.join(BASE_DIR, "monitoring", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_evidently_reports():
    print("Generating Evidently AI reports...")
    
    # Load training data (reference) and current data (test)
    train_path = os.path.join(BASE_DIR, "training", "training_dataset.parquet")
    if not os.path.exists(train_path):
        print("Training dataset not found.")
        return

    full_data = pd.read_parquet(train_path)
    
    # Split for demo: first 10k as reference, last 10k as production
    reference_data = full_data.head(10000)
    current_data = full_data.tail(10000)
    
    # Rename target for Evidently ClassificationPreset
    reference_data = reference_data.rename(columns={'label': 'target'})
    current_data = current_data.rename(columns={'label': 'target'})
    
    # Add dummy predictions for model performance report
    reference_data['prediction'] = np.random.uniform(0, 1, len(reference_data))
    current_data['prediction'] = np.random.uniform(0, 1, len(current_data))

    # 1. Data Drift Report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save_html(os.path.join(REPORT_DIR, "data_drift.html"))

    # 2. Target Drift Report
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference_data, current_data=current_data)
    target_drift_report.save_html(os.path.join(REPORT_DIR, "target_drift.html"))

    # 3. Data Quality Report
    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(reference_data=reference_data, current_data=current_data)
    data_quality_report.save_html(os.path.join(REPORT_DIR, "data_quality.html"))

    # 4. Model Performance Report
    model_perf_report = Report(metrics=[ClassificationPreset()])
    model_perf_report.run(reference_data=reference_data, current_data=current_data)
    model_perf_report.save_html(os.path.join(REPORT_DIR, "model_performance.html"))

    print(f"Evidently reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    generate_evidently_reports()
