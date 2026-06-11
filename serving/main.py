from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import os
import sys
import redis
import shap
import pandas as pd
import numpy as np

# Add parent path to sys.path for feature_store
sys.path.append(os.path.join(os.getcwd(), 'realtime-ml-platform'))
from feature_store.feature_store import FeatureStore
from serving.model_registry import ModelRegistry

app = FastAPI(title="Real-time ML Serving Platform")

# Initialize components
registry = ModelRegistry()
fs = FeatureStore()
r_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
redis_client = redis.Redis(connection_pool=r_pool)

# Mock model registry load on startup (real one would load from mlflow)
@app.on_event("startup")
async def startup_event():
    registry.load_latest_models()

class FeatureRequest(BaseModel):
    user_id: int
    item_ids: List[int]

class PredictRequest(BaseModel):
    user_id: int
    item_ids: List[int]

class RankingRequest(BaseModel):
    user_id: int
    candidate_item_ids: List[int]

@app.get("/health")
def health():
    try:
        redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "models_loaded": list(registry.models.keys()),
        "timestamp": time.time()
    }

@app.post("/features/online")
def get_online_features(req: FeatureRequest):
    start_time = time.time()
    try:
        # Fetch user features
        user_feats = fs.get_online_features('user_features', [req.user_id])
        # Fetch item features
        item_feats = fs.get_online_features('item_features', req.item_ids)
        
        latency = (time.time() - start_time) * 1000
        return {
            "user_features": user_feats.to_dict('records')[0],
            "item_features": item_feats.to_dict('records'),
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ctr")
def predict_ctr(req: PredictRequest):
    start_time = time.time()
    model = registry.get_model('ctr')
    if not model:
        # Fallback/Mock for demo
        probs = np.random.uniform(0, 0.1, len(req.item_ids)).tolist()
        return {"predictions": probs, "note": "Mock model used"}

    # Fetch features (Simplified for demo)
    user_feats = fs.get_online_features('user_features', [req.user_id]).iloc[0]
    item_feats = fs.get_online_features('item_features', req.item_ids)
    
    # Construct feature matrix
    X = item_feats.copy()
    for col in user_feats.index:
        X[col] = user_feats[col]
    
    # Predict
    preds = model.predict(X)
    
    # SHAP (Top 3 features)
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X.iloc[0:1])
    # top_features = ... (omitted for speed in demo)
    
    latency = (time.time() - start_time) * 1000
    return {
        "predictions": [
            {"item_id": iid, "score": float(score)} 
            for iid, score in zip(req.item_ids, preds)
        ],
        "latency_ms": latency
    }

@app.post("/predict/ranking")
def predict_ranking(req: RankingRequest):
    start_time = time.time()
    model = registry.get_model('ranking')
    
    # Simplify: If no model, just random rank
    if not model:
        top_10 = req.candidate_item_ids[:10]
        return {"ranked_items": top_10, "note": "Mock ranking"}
    
    # Ranking logic using XGBRanker...
    # Fetch features and rank
    latency = (time.time() - start_time) * 1000
    return {"ranked_items": req.candidate_item_ids[:10], "latency_ms": latency}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
