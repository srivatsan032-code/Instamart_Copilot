"""Part B tasks 1-3: RFM table, K-means, segment profile + PCA.  python src/part_b_segments.py"""
import numpy as np
import os
os.makedirs("outputs", exist_ok=True)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
o = o[o.status == "completed"]
g = o.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                 monetary=("order_amount", "sum"))
g["recency"] = (SNAPSHOT - g["last_ts"]).dt.days
rfm = g[["recency", "frequency", "monetary"]].reset_index()
print("=== RFM")
print("customers:", len(rfm))
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max())
print("monetary mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== K-means sweep (inertia, silhouette)")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"K={k}: inertia={km.inertia_:.1f} silhouette={silhouette_score(X, km.labels_):.3f}")

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)

# Names are assigned from the profile (cluster numbers are arbitrary labels with no order).
names = {}
premium_id = prof.mean_monetary.idxmax()
names[premium_id] = "Premium High-Spenders"
rest = prof.drop(index=premium_id)
dormant_id = rest.mean_recency.idxmax()
names[dormant_id] = "Dormant / At-Risk"
rest = rest.drop(index=dormant_id)
loyal_id = rest.mean_frequency.idxmax()
names[loyal_id] = "Loyal Regulars"
for sid in rest.index:
    names.setdefault(sid, "Casual Occasional Shoppers")
actions = {"Premium High-Spenders": "VIP early access to electronics launches and a dedicated premium support desk.",
           "Loyal Regulars": "Reward streaks with a loyalty/subscription offer and basket-building bundles.",
           "Casual Occasional Shoppers": "Nudge to a second weekly order with category coupons.",
           "Dormant / At-Risk": "Win-back offer with free delivery; stop discount spend after two attempts."}
prof["segment_name"] = prof.index.map(names)
prof["recommended_action"] = prof.segment_name.map(actions)
print("\n=== Segment profile (K=%d)" % FINAL_K)
print(prof.to_string())

rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium_id
rfm.round({"monetary": 2}).to_csv("outputs/customer_segments.csv", index=False)
print("\nPremium segment id:", premium_id, "| premium customers:", int(rfm.is_premium.sum()))

pca = PCA(n_components=2).fit(X)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))
