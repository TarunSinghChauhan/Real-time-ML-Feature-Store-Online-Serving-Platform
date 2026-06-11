import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.pytorch
import os

class ModelRegistry:
    def __init__(self, tracking_uri="sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.models = {}

    def load_latest_models(self):
        print("Loading latest models from MLflow...")
        try:
            # CTR Model
            ctr_run = mlflow.search_runs(experiment_names=["ctr_prediction"], order_by=["metrics.auc DESC"]).iloc[0]
            self.models['ctr'] = mlflow.lightgbm.load_model(f"runs:/{ctr_run.run_id}/lgb_ctr_model")
            
            # Ranking Model
            rank_run = mlflow.search_runs(experiment_names=["ranking_model"]).iloc[0]
            self.models['ranking'] = mlflow.xgboost.load_model(f"runs:/{rank_run.run_id}/xgb_ranker")
            
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Mock models for demo if loading fails
            self.models['ctr'] = None
            self.models['ranking'] = None

    def get_model(self, name):
        return self.models.get(name)

    def get_info(self, name):
        # In a real system, fetch from MLflow registry tags
        return {
            "name": name,
            "version": "1.0.0",
            "status": "Production"
        }
