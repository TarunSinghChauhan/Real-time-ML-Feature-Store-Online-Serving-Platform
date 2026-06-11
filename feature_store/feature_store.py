import pandas as pd
import numpy as np
import redis
import json
import os
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

class FeatureStore:
    def __init__(self, redis_host='localhost', redis_port=6379, offline_path='feature_store/offline'):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.offline_path = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform" + "\\" + offline_path
        self.feature_views = {}
        os.makedirs(self.offline_path, exist_ok=True)

    def register_feature_view(self, name, entity, features, ttl_seconds=None):
        self.feature_views[name] = {
            'entity': entity,
            'features': features,
            'ttl': ttl_seconds
        }
        print(f"Registered Feature View: {name}")

    def materialize_offline(self, feature_view_name, feature_df, timestamp_col='event_timestamp'):
        """
        Stores historical features as Parquet files.
        """
        if feature_view_name not in self.feature_views:
            raise ValueError(f"Feature view {feature_view_name} not registered.")
        
        path = os.path.join(self.offline_path, f"{feature_view_name}.parquet")
        table = pa.Table.from_pandas(feature_df)
        pq.write_table(table, path)
        print(f"Materialized Offline: {feature_view_name} to {path}")

    def materialize_online(self, feature_view_name, feature_df):
        """
        Writes features to Redis with TTL.
        """
        if feature_view_name not in self.feature_views:
            raise ValueError(f"Feature view {feature_view_name} not registered.")
        
        entity_col = self.feature_views[feature_view_name]['entity']
        ttl = self.feature_views[feature_view_name]['ttl']
        
        pipeline = self.redis.pipeline()
        for _, row in feature_df.iterrows():
            entity_id = row[entity_col]
            # Convert row to dict, excluding timestamp or metadata if necessary
            data = row.to_dict()
            key = f"{feature_view_name}:{entity_id}"
            pipeline.hset(key, mapping={k: str(v) for k, v in data.items()})
            if ttl:
                pipeline.expire(key, ttl)
        
        pipeline.execute()
        print(f"Materialized Online: {feature_view_name} to Redis")

    def get_online_features(self, feature_view_name, entity_ids):
        """
        Real-time feature retrieval from Redis.
        """
        results = []
        for eid in entity_ids:
            key = f"{feature_view_name}:{eid}"
            data = self.redis.hgetall(key)
            if data:
                # Convert back from strings
                results.append(data)
            else:
                results.append({})
        return pd.DataFrame(results)

    def get_offline_features(self, entity_df, feature_view_names, join_keys):
        """
        Point-in-time correct historical feature retrieval.
        entity_df: [entity_id, event_timestamp, ...]
        """
        result_df = entity_df.copy()
        
        for feature_view_name in feature_view_names:
            path = os.path.join(self.offline_path, f"{feature_view_name}.parquet")
            if not os.path.exists(path):
                print(f"Warning: Offline storage for {feature_view_name} not found.")
                continue
            
            features_df = pd.read_parquet(path)
            entity_key = self.feature_views[feature_view_name]['entity']
            
            # Point-in-time join logic: 
            # For each row in entity_df, find the latest feature_df record where 
            # feature_timestamp <= entity_timestamp and entity_id matches.
            # This is complex in pure pandas for large datasets. Using merge_asof.
            
            # Ensure timestamps are datetime
            result_df['event_timestamp'] = pd.to_datetime(result_df['event_timestamp'])
            features_df['feature_timestamp'] = pd.to_datetime(features_df['feature_timestamp'])
            
            # Sort for merge_asof
            result_df = result_df.sort_values('event_timestamp')
            features_df = features_df.sort_values('feature_timestamp')
            
            result_df = pd.merge_asof(
                result_df,
                features_df,
                left_on='event_timestamp',
                right_on='feature_timestamp',
                by=entity_key,
                direction='backward'
            )
            
        return result_df

    def get_feature_statistics(self, feature_view_name):
        """
        Computes distribution statistics for drift monitoring.
        """
        path = os.path.join(self.offline_path, f"{feature_view_name}.parquet")
        if not os.path.exists(path):
            return {}
        
        df = pd.read_parquet(path)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'quantiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Save stats to JSON
        stats_path = os.path.join(self.offline_path, f"{feature_view_name}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
            
        return stats
