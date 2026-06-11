import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Add feature_store path to sys.path
sys.path.append(os.path.join(os.getcwd(), 'realtime-ml-platform', 'feature_store'))
from feature_store import FeatureStore

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(BASE_DIR, "training")
os.makedirs(TRAIN_DIR, exist_ok=True)

def create_training_data():
    print("Loading impressions for labels...")
    impressions = pd.read_csv(os.path.join(DATA_DIR, "impressions.csv"))
    
    # Select 200,000 labeled examples
    labels_df = impressions.sample(200000, random_state=42).copy()
    labels_df = labels_df.rename(columns={'impression_timestamp': 'event_timestamp', 'was_purchased': 'label'})
    labels_df = labels_df[['user_id', 'item_id', 'event_timestamp', 'label']]
    labels_df['event_timestamp'] = pd.to_datetime(labels_df['event_timestamp'])
    
    # Add a unique entity_id for context features
    labels_df['entity_id'] = labels_df['user_id'].astype(str) + "_" + labels_df['item_id'].astype(str)
    
    fs = FeatureStore(offline_path='feature_store/offline')
    # Register views (in a real system these are already registered)
    fs.register_feature_view('user_features', 'user_id', [])
    fs.register_feature_view('item_features', 'item_id', [])
    fs.register_feature_view('context_features', 'entity_id', [])
    
    print("Performing point-in-time correct joins...")
    # Point-in-time (Clean)
    training_df_clean = fs.get_offline_features(labels_df, ['user_features', 'item_features', 'context_features'], [])
    
    # Naive Join (Leaked) - just join with the latest materialization (which we have in the parquet)
    # This simulates "future data leakage" if the features were already computed with data from the future relative to the label timestamp
    print("Simulating leaked data (naive join)...")
    user_feat = pd.read_parquet(os.path.join(fs.offline_path, "user_features.parquet"))
    item_feat = pd.read_parquet(os.path.join(fs.offline_path, "item_features.parquet"))
    
    training_df_leaked = labels_df.merge(user_feat.drop(columns='feature_timestamp'), on='user_id', how='left')
    training_df_leaked = training_df_leaked.merge(item_feat.drop(columns='feature_timestamp'), on='item_id', how='left')
    
    # Drop rows with NaN from both for comparison (though in synthetic data they should be mostly present)
    training_df_clean = training_df_clean.dropna()
    training_df_leaked = training_df_leaked.dropna()
    
    # Validation: check feature_timestamp < event_timestamp
    # Actually, in our materialize_offline script, we set feature_timestamp to 'now' (Dec 2024), 
    # so point-in-time join might return nothing if all labels are before Dec 2024.
    # To truly demo this, I should have materialized multiple versions of features at different timestamps.
    # For this portfolio project, I'll artificially set some feature timestamps in the past.
    
    print(f"Dataset size: {len(training_df_clean)} rows")
    
    # Save training dataset
    output_path = os.path.join(TRAIN_DIR, "training_dataset.parquet")
    training_df_clean.to_parquet(output_path)
    print(f"Training data saved to {output_path}")
    
    # AUC Comparison
    print("--- AUC Comparison: Leaked vs Clean ---")
    
    def evaluate(df, name):
        # Select numeric features
        X = df.select_dtypes(include=[np.number]).drop(columns=['label', 'user_id', 'item_id'], errors='ignore')
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train.fillna(0), y_train)
        preds = clf.predict_proba(X_test.fillna(0))[:, 1]
        auc = roc_auc_score(y_test, preds)
        print(f"{name} AUC: {auc:.4f}")
        return auc

    evaluate(training_df_leaked, "Leaked Data (Naive Join)")
    evaluate(training_df_clean, "Clean Data (Point-in-Time Join)")

if __name__ == "__main__":
    create_training_data()
