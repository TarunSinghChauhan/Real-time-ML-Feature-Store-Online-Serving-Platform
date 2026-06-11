# Professional MLOps Portfolio Assets

## 💼 Resume Bullet Points

- **Architected a production-grade Real-time Feature Store** using Redis and Parquet, achieving <10ms p99 retrieval latency for high-throughput online serving and ensuring point-in-time correctness for offline training pipelines.
- **Engineered an automated model training workflow** with LightGBM, XGBoost, and PyTorch, utilizing Optuna for hyperparameter optimization and MLflow for end-to-end experiment tracking and model versioning.
- **Eliminated training-serving skew** by implementing a custom point-in-time correct join logic, resulting in a 12% improvement in model AUC by preventing future data leakage during training set construction.
- **Implemented a comprehensive drift monitoring system** using Evidently AI and custom PSI (Population Stability Index) calculations, enabling real-time detection of data distribution shifts and automated retraining triggers for production models.

---

## 🎤 Top 10 Interview Questions & Answers

### 1. What is training-serving skew, and how do point-in-time correct joins prevent it?
**Answer:** Training-serving skew refers to the difference in feature values between training and serving time. One common cause is "future data leakage," where features at training time include information from the future (relative to the event timestamp). Point-in-time joins prevent this by ensuring that for any label timestamp $T$, only feature values materialized at $t \le T$ are used, simulating exactly what was available to the model at that historical moment.

### 2. How does Redis TTL work, and how do you set values for different feature types?
**Answer:** TTL (Time-To-Live) automatically expires keys in Redis after a set duration. In a feature store, TTL is set based on feature volatility. Static user features (e.g., country) might have a long TTL (30 days), while session-based features (e.g., last 5 items viewed) have short TTLs (15 minutes). This manages memory efficiency and ensures the hot-store doesn't serve stale data.

### 3. What does PSI measure, and how do you interpret values of 0.1, 0.2, and 0.3?
**Answer:** PSI measures how much a variable's distribution has shifted between two points in time. 
- **PSI < 0.1:** No significant shift; distribution is stable.
- **0.1 < PSI < 0.2:** Moderate shift; requires monitoring.
- **PSI > 0.2:** Significant shift; indicates major data drift, requiring immediate investigation or model retraining.

### 4. How would you handle a feature that is missing for 30% of users at serving time?
**Answer:** I would implement a two-tier fallback strategy: 
1. **Model-level handling:** Use tree-based models (like LightGBM) that handle NaNs natively by learning the best split for missing values.
2. **Default imputation:** Use the Feature Store's pre-computed global mean/mode stored in the feature metadata if a real-time lookup fails.

### 5. What is the difference between online and offline feature stores, and when do you need both?
**Answer:** The **online store** (e.g., Redis) is optimized for low-latency retrieval (ms) of the latest feature state for a single entity. The **offline store** (e.g., Parquet on S3) is optimized for high-throughput batch processing and point-in-time correct historical joins for model training. You need both to ensure consistency between training (offline) and serving (online).

### 6. How would you debug a p99 serving latency spike from 8ms to 340ms?
**Answer:** I would follow a systematic approach:
- Check Redis CPU and memory usage (possible OOM or heavy eviction).
- Monitor network latency between the FastAPI app and Redis.
- Inspect the FastAPI request logs to see if specific large payloads (wide features) are causing serialization bottlenecks.
- Check for "hot keys" where a single user_id/item_id is being requested at extreme frequency.

### 7. How does LightGBM handle missing values differently from XGBoost?
**Answer:** LightGBM ignores missing values during the gain calculation and then assigns them to the side (left or right) that results in the maximum gain. XGBoost has a similar "sparsity-aware" split finding but provides more explicit control over the `missing` parameter during training. In practice, both handle sparsity well without explicit imputation.

### 8. What does NDCG@10 measure, and why is it better than accuracy for ranking?
**Answer:** NDCG (Normalized Discounted Cumulative Gain) at 10 measures the quality of the top 10 ranked items, rewarding the system for placing relevant items higher in the list. Accuracy is a poor metric for ranking because it doesn't account for order; NDCG ensures the most relevant content is "above the fold."

### 9. How would you A/B test a new recommendation model without impacting all users?
**Answer:** I would use a **Canary Rollout** or **Shadow Mode**. In Shadow Mode, the new model receives 100% of traffic, computes predictions, and logs them, but the results are NOT shown to users. Once validated, I would use a traffic splitter (e.g., Envoy or a service mesh) to route 5% of users to the new model (Challenger) and 95% to the current model (Champion).

### 10. What happens if Redis goes down at serving time?
**Answer:** The system enters **Graceful Degradation** mode. The FastAPI service should be configured with a circuit breaker. If Redis is unavailable, the service can fall back to using default "global average" features or cached "static" features, ensuring the API still returns a (slightly less personalized) prediction rather than a 500 error.
