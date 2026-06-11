import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 50000
NUM_ITEMS = 10000
NUM_ORDERS = 500000
NUM_SESSIONS = 2000000
NUM_IMPRESSIONS = 5000000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
START_TS = int(START_DATE.timestamp())
END_TS = int(END_DATE.timestamp())

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def random_dates(start_ts, end_ts, n):
    ts = np.random.randint(start_ts, end_ts, n)
    return pd.to_datetime(ts, unit='s')

print("--- Generating Users ---")
user_ids = np.arange(1, NUM_USERS + 1)
countries = ['US', 'UK', 'CA', 'DE', 'FR', 'IN', 'JP', 'BR']
device_types = ['mobile', 'desktop', 'tablet']
plan_tiers = ['free', 'silver', 'gold', 'platinum']
acquisition_channels = ['organic', 'paid_search', 'social_media', 'referral', 'email']
age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']

users_df = pd.DataFrame({
    'user_id': user_ids,
    'signup_date': random_dates(START_TS, END_TS, NUM_USERS),
    'country': np.random.choice(countries, NUM_USERS),
    'device_type': np.random.choice(device_types, NUM_USERS, p=[0.6, 0.3, 0.1]),
    'plan_tier': np.random.choice(plan_tiers, NUM_USERS, p=[0.5, 0.3, 0.15, 0.05]),
    'acquisition_channel': np.random.choice(acquisition_channels, NUM_USERS),
    'age_group': np.random.choice(age_groups, NUM_USERS)
})
users_df.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)

print("--- Generating Items ---")
item_ids = np.arange(1, NUM_ITEMS + 1)
categories = ['Electronics', 'Home & Kitchen', 'Fashion', 'Beauty', 'Sports', 'Toys']
item_cats = np.random.choice(categories, NUM_ITEMS)

prices = np.random.lognormal(mean=3.5, sigma=1.0, size=NUM_ITEMS).clip(5, 5000)

items_df = pd.DataFrame({
    'item_id': item_ids,
    'category': item_cats,
    'price': prices.round(2),
    'brand': [f"Brand_{i}" for i in np.random.randint(1, 500, NUM_ITEMS)],
    'avg_rating': np.random.uniform(3.0, 5.0, NUM_ITEMS).round(1),
    'review_count': np.random.poisson(lam=50, size=NUM_ITEMS),
    'days_since_listed': np.random.randint(1, 1000, NUM_ITEMS)
})
items_df.to_csv(os.path.join(DATA_DIR, "items.csv"), index=False)

item_ranks = np.arange(1, NUM_ITEMS + 1)
item_weights = 1.0 / (item_ranks ** 0.8)
item_weights /= item_weights.sum()

print("--- Generating Sessions ---")
session_starts = random_dates(START_TS, END_TS, NUM_SESSIONS)
session_lengths = np.random.lognormal(mean=np.log(600), sigma=0.8, size=NUM_SESSIONS)
session_ends = session_starts + pd.to_timedelta(session_lengths, unit='s')

sessions_df = pd.DataFrame({
    'session_id': np.arange(1, NUM_SESSIONS + 1),
    'user_id': np.random.choice(user_ids, NUM_SESSIONS),
    'session_start': session_starts,
    'session_end': session_ends,
    'pages_viewed': np.random.poisson(lam=5, size=NUM_SESSIONS) + 1,
    'device_type': np.random.choice(device_types, NUM_SESSIONS),
    'referrer': np.random.choice(['google', 'direct', 'facebook', 'instagram', 'ad_network'], NUM_SESSIONS)
})
sessions_df['items_viewed'] = (sessions_df['pages_viewed'] * np.random.uniform(0.4, 0.8, NUM_SESSIONS)).astype(int)
sessions_df['items_added_to_cart'] = (sessions_df['items_viewed'] * np.random.uniform(0.05, 0.2, NUM_SESSIONS)).astype(int)
sessions_df.to_csv(os.path.join(DATA_DIR, "sessions.csv"), index=False)

print("--- Generating Impressions ---")
impression_timestamps = random_dates(START_TS, END_TS, NUM_IMPRESSIONS)
was_clicked = np.random.choice([0, 1], NUM_IMPRESSIONS, p=[0.958, 0.042])
was_purchased = np.zeros(NUM_IMPRESSIONS, dtype=int)
clicked_indices = np.where(was_clicked == 1)[0]
was_purchased[clicked_indices] = np.random.choice([0, 1], len(clicked_indices), p=[0.81, 0.19])

impressions_df = pd.DataFrame({
    'impression_id': np.arange(1, NUM_IMPRESSIONS + 1),
    'user_id': np.random.choice(user_ids, NUM_IMPRESSIONS),
    'item_id': np.random.choice(item_ids, NUM_IMPRESSIONS, p=item_weights),
    'impression_timestamp': impression_timestamps,
    'position': np.random.randint(1, 20, NUM_IMPRESSIONS),
    'was_clicked': was_clicked,
    'was_purchased': was_purchased,
    'recommendation_source': np.random.choice(['trending', 'personalized', 'collaborative', 'search'], NUM_IMPRESSIONS)
})
impressions_df.to_csv(os.path.join(DATA_DIR, "impressions.csv"), index=False)

print("--- Generating Orders ---")
purchase_impressions = impressions_df[impressions_df['was_purchased'] == 1].copy()
num_purchase_from_impressions = len(purchase_impressions)
remaining_orders = NUM_ORDERS - num_purchase_from_impressions

if remaining_orders > 0:
    extra_orders = pd.DataFrame({
        'user_id': np.random.choice(user_ids, remaining_orders),
        'item_id': np.random.choice(item_ids, remaining_orders, p=item_weights),
        'order_timestamp': random_dates(START_TS, END_TS, remaining_orders)
    })
else:
    extra_orders = pd.DataFrame(columns=['user_id', 'item_id', 'order_timestamp'])

order_data_full = pd.concat([
    purchase_impressions[['user_id', 'item_id', 'impression_timestamp']].rename(columns={'impression_timestamp': 'order_timestamp'}),
    extra_orders
])

order_data_full['order_id'] = np.arange(1, len(order_data_full) + 1)
order_data_full = order_data_full.merge(items_df[['item_id', 'category', 'price']], on='item_id', how='left')
order_data_full['order_value'] = order_data_full['price']
order_data_full['payment_method'] = np.random.choice(['credit_card', 'paypal', 'apple_pay', 'crypto'], len(order_data_full))
order_data_full['status'] = np.random.choice(['completed', 'cancelled', 'returned'], len(order_data_full), p=[0.9, 0.05, 0.05])

order_data_full = order_data_full[['order_id', 'user_id', 'item_id', 'order_timestamp', 'order_value', 'category', 'payment_method', 'status']]
order_data_full.to_csv(os.path.join(DATA_DIR, "orders.csv"), index=False)

print("Data generation complete.")
