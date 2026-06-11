# Real-time ML Feature Store + Online Serving Platform 🚀

A production-grade, end-to-end MLOps platform demonstrating a complete feature store architecture, point-in-time correct training pipelines, multi-model serving, and automated drift monitoring.

## 🏗️ Architecture Overview

```ascii
[ Synthetic Data ] --> [ Feature Store (Offline/Online) ] --> [ MLflow Lifecycle ]
       |                         |                              |
       v                         v                              v
[ User Behavior ]      [ Materialization Jobs ]         [ Model Registry ]
[ Order History ]      [ Point-in-Time Join ]           [ Staging/Prod ]
       |                         |                              |
       +-------------------------+------------------------------+
                                 |
                                 v
                       [ FastAPI Serving App ] <--- [ Redis (Hot Features) ]
                                 |
                                 v
                       [ Streamlit Dashboard ] <--- [ Evidently AI Drift Reports ]
```

## 🛠️ Tech Stack

- **Feature Store**: Redis (Online), Parquet (Offline), Python
- **Model Training**: LightGBM (CTR), XGBoost (Ranking), PyTorch (NCF), Optuna
- **Experiment Tracking**: MLflow
- **Serving**: FastAPI, Redis Connection Pooling, Pydantic
- **Monitoring**: Evidently AI, Plotly, PSI calculations
- **Infrastructure**: Docker, Docker Compose, Makefile

## 📁 Project Structure

- `/data`: Synthetic data generation logic (5M+ rows).
- `/feature_store`: `FeatureStore` class and materialization scripts.
- `/training`: Point-in-time correct dataset pipeline and model training.
- `/serving`: FastAPI application and model registry.
- `/monitoring`: Drift detection, Evidently AI reports, and Streamlit dashboard.
- `/docker`: Containerization setup and `docker-compose`.

## 🚀 Quick Start

### 1. Local Setup
```bash
# Generate 5M+ records of synthetic data
make generate-data

# Compute and materialize features
make compute-features

# Create point-in-time correct training data
make train-data

# Train all models (LGBM, XGB, PyTorch) via MLflow
make train
```

### 2. Docker Execution
```bash
# Start the entire platform (Redis, MLflow, FastAPI, Streamlit)
make docker-up
```

## 📊 API Documentation

### POST `/predict/ctr`
Predict Click-Through Rate for a list of candidate items.
```bash
curl -X POST "http://localhost:8000/predict/ctr" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1234, "item_ids": [101, 102, 105]}'
```

### POST `/features/online`
Retrieve real-time features from Redis Feature Store.
```bash
curl -X POST "http://localhost:8000/features/online" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1234, "item_ids": [101]}'
```

## 📈 Monitoring & Drift
The system automatically monitors for feature drift using **Population Stability Index (PSI)**. If more than 3 features show significant drift (PSI > 0.2), an automated retraining trigger is logged. Detailed reports are available in the Streamlit dashboard at `http://localhost:8501`.

---
*Created by Tarun*
