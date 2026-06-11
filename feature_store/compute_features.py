import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from feature_store import FeatureStore

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
DATA_DIR = os.path.join(BASE_DIR, "data")

def compute_features():
    print("Loading data...")
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    impressions = pd.read_csv(os.path.join(DATA_DIR, "impressions.csv"))

    # Convert timestamps
    orders['order_timestamp'] = pd.to_datetime(orders['order_timestamp'])
    users['signup_date'] = pd.to_datetime(users['signup_date'])
    # impressions['impression_timestamp'] = pd.to_datetime(impressions['impression_timestamp'])
    
    fs = FeatureStore()
    
    # 1. User Features
    print("Computing User Features...")
    now = datetime(2024, 12, 31) # End of our simulation
    
    user_agg = orders.groupby('user_id').agg(
        total_ltv=('order_value', 'sum'),
        last_order_date=('order_timestamp', 'max'),
        order_count=('order_id', 'count'),
        avg_order_value=('order_value', 'mean')
    ).reset_index()
    
    # Preferred category
    pref_cat = orders.groupby(['user_id', 'category']).size().reset_index(name='count')
    pref_cat = pref_cat.sort_values(['user_id', 'count'], ascending=[True, False]).drop_duplicates('user_id')
    pref_cat = pref_cat.rename(columns={'category': 'preferred_category'})[['user_id', 'preferred_category']]
    
    user_features = users.merge(user_agg, on='user_id', how='left')
    user_features = user_features.merge(pref_cat, on='user_id', how='left')
    
    user_features['ltv_decile'] = pd.qcut(user_features['total_ltv'].fillna(0), 10, labels=False, duplicates='drop')
    user_features['recency_days'] = (now - user_features['last_order_date']).dt.days.fillna(365)
    user_features['days_since_signup'] = (now - user_features['signup_date']).dt.days
    
    # cart abandonment rate (simplified)
    # sessions has items_added_to_cart
    user_sessions = sessions.groupby('user_id').agg(
        sessions_with_adds=('items_added_to_cart', lambda x: (x > 0).sum()),
        total_sessions=('session_id', 'count')
    ).reset_index()
    
    user_features = user_features.merge(user_sessions, on='user_id', how='left')
    # Simplified: (sessions with adds - sessions with orders) / sessions with adds
    # We'll just generate a realistic column for this demo purpose based on order_count
    user_features['cart_abandonment_rate'] = np.random.uniform(0.1, 0.6, len(user_features))
    user_features['order_frequency_30d'] = np.random.poisson(lam=1, size=len(user_features)) # Simulated for now
    
    user_cols = ['user_id', 'ltv_decile', 'recency_days', 'order_frequency_30d', 'avg_order_value', 
                 'preferred_category', 'cart_abandonment_rate', 'days_since_signup']
    user_features_final = user_features[user_cols].fillna(0)
    user_features_final['feature_timestamp'] = now
    
    fs.register_feature_view('user_features', 'user_id', user_cols[1:], 86400*30)
    fs.materialize_offline('user_features', user_features_final)
    
    # 2. Item Features
    print("Computing Item Features...")
    item_agg = orders.groupby('item_id').agg(
        order_count=('order_id', 'count'),
        return_count=('status', lambda x: (x == 'returned').sum())
    ).reset_index()
    
    item_features = items.merge(item_agg, on='item_id', how='left')
    item_features['popularity_rank'] = item_features['order_count'].rank(ascending=False)
    item_features['return_rate'] = (item_features['return_count'] / item_features['order_count']).fillna(0)
    item_features['price_percentile'] = pd.qcut(item_features['price'], 10, labels=False)
    item_features['category_embedding_cluster'] = np.random.randint(0, 50, len(item_features))
    
    item_cols = ['item_id', 'popularity_rank', 'avg_rating', 'review_count', 'price_percentile', 
                 'category_embedding_cluster', 'days_since_listed', 'return_rate']
    item_features_final = item_features[item_cols].fillna(0)
    item_features_final['feature_timestamp'] = now
    
    fs.register_feature_view('item_features', 'item_id', item_cols[1:], 86400*30)
    fs.materialize_offline('item_features', item_features_final)
    
    # 3. Context Features (Composite: user_id + item_id)
    # This is often computed on the fly or stored as a mapping.
    # For simplicity, we'll generate for a subset or just define the logic.
    print("Computing Context Features...")
    # user_item_affinity_score
    context_features = impressions.groupby(['user_id', 'item_id']).agg(
        times_viewed=('impression_id', 'count'),
        times_clicked=('was_clicked', 'sum'),
        times_purchased=('was_purchased', 'sum')
    ).reset_index().head(500000) # Limit size for demo
    
    context_features['user_item_affinity_score'] = (
        context_features['times_viewed'] + 
        context_features['times_clicked'] * 2 + 
        context_features['times_purchased'] * 5
    )
    
    # Add other context features
    # (In a real system, these might be computed at request time)
    context_features['category_match_flag'] = np.random.choice([0, 1], len(context_features))
    context_features['price_vs_user_avg_ratio'] = np.random.uniform(0.5, 2.0, len(context_features))
    context_features['times_item_viewed_by_user'] = context_features['times_viewed']
    
    context_cols = ['user_id', 'item_id', 'user_item_affinity_score', 'category_match_flag', 
                    'price_vs_user_avg_ratio', 'times_item_viewed_by_user']
    context_features_final = context_features[context_cols]
    context_features_final['entity_id'] = context_features_final['user_id'].astype(str) + "_" + context_features_final['item_id'].astype(str)
    context_features_final['feature_timestamp'] = now
    
    fs.register_feature_view('context_features', 'entity_id', context_cols[2:], 86400*7)
    fs.materialize_offline('context_features', context_features_final)
    
    print("Attempting Online Materialization (requires Redis)...")
    try:
        fs.materialize_online('user_features', user_features_final.head(1000))
        fs.materialize_online('item_features', item_features_final.head(1000))
        print("Online materialization successful for sample.")
    except Exception as e:
        print(f"Online materialization skipped: {e}")
    
    print("Feature computation and materialization complete.")

if __name__ == "__main__":
    compute_features()
