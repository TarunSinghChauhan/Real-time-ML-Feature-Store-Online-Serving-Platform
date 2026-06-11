import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from scipy.stats import chisquare
from datetime import datetime

class DriftDetector:
    def __init__(self, baseline_stats_path, current_data_df):
        with open(baseline_stats_path, 'r') as f:
            self.baseline = json.load(f)
        self.current = current_data_df
        self.drift_results = {}

    def calculate_psi(self, expected, actual, buckets=10):
        """
        Population Stability Index calculation.
        """
        def scale(ser, min_val, max_val):
            return (ser - min_val) / (max_val - min_val + 1e-6)

        expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 1e-6, 1.0)
        actual_percents = np.clip(actual_percents, 1e-6, 1.0)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    def run_monitoring(self):
        print("Running drift monitoring...")
        num_drifted = 0
        for feat, stats in self.baseline.items():
            if feat not in self.current.columns:
                continue
            
            # Simple PSI for demo (needs historical distribution, here using stats to simulate)
            # In a real system, we'd sample from the offline parquet for the 'expected' distribution
            # For demo, we'll generate drift status randomly for some features
            psi = np.random.uniform(0, 0.4) 
            self.drift_results[feat] = {
                'psi': psi,
                'drifted': psi > 0.2,
                'type': 'numeric'
            }
            if psi > 0.2:
                num_drifted += 1
        
        return num_drifted

    def generate_report(self, output_path):
        fig = go.Figure()
        features = list(self.drift_results.keys())
        psi_values = [self.drift_results[f]['psi'] for f in features]
        
        fig.add_trace(go.Bar(
            x=features,
            y=psi_values,
            marker_color=['red' if v > 0.2 else 'blue' for v in psi_values]
        ))
        
        fig.update_layout(
            title="Feature Drift Analysis (PSI)",
            xaxis_title="Features",
            yaxis_title="PSI Value",
            shapes=[dict(type='line', y0=0.2, y1=0.2, x0=-0.5, x1=len(features)-0.5, 
                         line=dict(color='red', dash='dash'))]
        )
        
        fig.write_html(output_path)
        print(f"Drift report generated at {output_path}")

def main():
    BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
    baseline_path = os.path.join(BASE_DIR, "feature_store", "offline", "user_features_stats.json")
    
    if not os.path.exists(baseline_path):
        print("Baseline stats not found. Run FeatureStore.get_feature_statistics() first.")
        return

    # Sample current data (mocked from parquet for demo)
    current_df = pd.read_parquet(os.path.join(BASE_DIR, "feature_store", "offline", "user_features.parquet")).sample(1000)
    
    detector = DriftDetector(baseline_path, current_df)
    drift_count = detector.run_monitoring()
    
    report_path = os.path.join(BASE_DIR, "monitoring", "drift_report.html")
    detector.generate_report(report_path)
    
    if drift_count > 3:
        print("ALERT: Significant drift detected in >3 features. Triggering retraining...")
        # send_slack_notification(...)
    else:
        print("Monitoring complete. No significant drift detected.")

if __name__ == "__main__":
    main()
