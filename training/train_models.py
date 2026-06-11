import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pytorch
import optuna
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, ndcg_score
from sklearn.model_selection import TimeSeriesSplit
import shap
import json

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "training", "training_dataset.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "training", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# --- 1. LightGBM CTR Model ---
def train_lightgbm(X, y, groups):
    print("Training LightGBM CTR Model...")
    mlflow.set_experiment("ctr_prediction")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.train(params, lgb.Dataset(X_t, label=y_t))
            preds = model.predict(X_v)
            aucs.append(roc_auc_score(y_v, preds))
        
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Reduced for demo speed
    
    best_params = study.best_params
    best_params['objective'] = 'binary'
    
    with mlflow.start_run():
        mlflow.log_params(best_params)
        model = lgb.train(best_params, lgb.Dataset(X, label=y))
        
        # Log artifacts
        mlflow.lightgbm.log_model(model, "lgb_ctr_model")
        mlflow.log_metric("auc", study.best_value)
        
        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.sample(100))
        # Log summary plot (simplified for demo)
        print(f"LGBM AUC: {study.best_value}")

# --- 2. XGBoost Ranking Model ---
def train_xgboost_ranker(X, y, groups):
    print("Training XGBoost Ranker...")
    mlflow.set_experiment("ranking_model")
    
    # Sort by group for XGBRanker
    # Grouping by user_id for ranking
    ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        lambdarank_pair_method='topk',
        eval_metric='ndcg@10',
        n_estimators=100
    )
    
    with mlflow.start_run():
        # group is the count of items per user
        group_counts = X.groupby('user_id').size().values
        X_rank = X.sort_values('user_id').drop(columns=['user_id', 'item_id'], errors='ignore')
        y_rank = y.iloc[X_rank.index]
        
        ranker.fit(X_rank, y_rank, group=group_counts)
        mlflow.xgboost.log_model(ranker, "xgb_ranker")
        print("XGBRanker trained.")

# --- 3. Neural CF Model (PyTorch) ---
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size=16):
        super(NCF, self).__init__()
        self.user_embed = nn.Embedding(num_users + 1, embed_size)
        self.item_embed = nn.Embedding(num_items + 1, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_idx, item_idx):
        u = self.user_embed(user_idx)
        i = self.item_embed(item_idx)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x)

def train_pytorch(X, y):
    print("Training PyTorch NCF Model...")
    mlflow.set_experiment("collaborative_filtering")
    
    num_users = int(X['user_id'].max())
    num_items = int(X['item_id'].max())
    model = NCF(num_users, num_items)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    user_tensor = torch.LongTensor(X['user_id'].values)
    item_tensor = torch.LongTensor(X['item_id'].values)
    label_tensor = torch.FloatTensor(y.values).view(-1, 1)
    
    with mlflow.start_run():
        for epoch in range(2): # Reduced for demo
            model.train()
            optimizer.zero_grad()
            outputs = model(user_tensor, item_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=epoch)
            
        mlflow.pytorch.log_model(model, "ncf_model")
        print("PyTorch NCF trained.")

def main():
    if not os.path.exists(TRAIN_DATA_PATH):
        print("Training data not found. Run create_training_data.py first.")
        return
        
    df = pd.read_parquet(TRAIN_DATA_PATH)
    df = df.sort_values('event_timestamp')
    
    # Drop non-numeric for LGBM/XGB (keep user_id for ranking group)
    y = df['label']
    X = df.drop(columns=['label', 'event_timestamp', 'entity_id', 'feature_timestamp'], errors='ignore')
    
    # Preprocessing: Convert preferred_category to codes
    if 'preferred_category' in X.columns:
        X['preferred_category'] = X['preferred_category'].astype('category').cat.codes
    
    train_lightgbm(X.drop(columns=['user_id', 'item_id'], errors='ignore'), y, None)
    train_xgboost_ranker(X, y, None)
    train_pytorch(X, y)

if __name__ == "__main__":
    main()
