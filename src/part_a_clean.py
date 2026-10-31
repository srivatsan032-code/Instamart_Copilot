"""Part A: audit raw exports, clean them with explicit rules, store as CSV + SQLite.
Run from repo root:  python src/part_a_clean.py"""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)

cust_raw = pd.read_csv(f"{RAW}/customers.csv")
prod_raw = pd.read_csv(f"{RAW}/products.csv")
ord_raw = pd.read_csv(f"{RAW}/orders.csv")

# ------------------------------------------------------------------ audit
def audit(name, df):
    print(f"\n=== {name}: shape={df.shape}")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())

for n, d in [("customers", cust_raw), ("products", prod_raw), ("orders", ord_raw)]:
    audit(n, d)

amt_num = pd.to_numeric(ord_raw["order_amount"], errors="coerce")
audit_rows = {
    "raw orders rows": len(ord_raw),
    "exact duplicate rows": int(ord_raw.duplicated().sum()),
    "rows with status pending": int((ord_raw["status"] == "pending").sum()),
    "distinct order_id": int(ord_raw["order_id"].nunique()),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((ord_raw["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(ord_raw["delivery_minutes"].isna().sum()),
    "customers: blank age": int(cust_raw["age"].isna().sum()),
    "customers: age outside 10-100": int(((cust_raw["age"] < 10) | (cust_raw["age"] > 100)).sum()),
    "customers: distinct raw city spellings": int(cust_raw["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(cust_raw["gender"].dropna().nunique()),
    "customers: blank gender": int(cust_raw["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES")
for k, v in audit_rows.items():
    print(f"{k}: {v}")

# ------------------------------------------------------------------ clean customers
c = cust_raw.copy()
c["city"] = c["city"].str.strip().str.title()                                  # (a)
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}        # (b)
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
bad_age = (c["age"] < 10) | (c["age"] > 100)                                   # (c)
c.loc[bad_age, "age"] = np.nan
n_age_missing = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])                            # (d)
print("\n=== CLEAN CUSTOMERS")
print("rows:", len(c), "| ages set to missing (out of range):", int(bad_age.sum()),
      "| ages filled by city median:", n_age_missing)
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print("missing ages:", int(c["age"].isna().sum()), "min/max age:", c["age"].min(), c["age"].max())

# ------------------------------------------------------------------ clean orders
o = ord_raw.copy()
steps = {}
o["order_ts"] = pd.to_datetime(o["order_ts"])                                   # rule 1
n0 = len(o)
o = o.drop_duplicates()                                                         # rule 2
steps["rule 2: exact duplicate rows removed"] = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # rule 3
steps["rule 3: superseded earlier records removed"] = n1 - len(o)
o = o.merge(prod_raw[["product_id", "category", "price_inr"]], on="product_id", how="left")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")           # rule 4
miss_amt = o["order_amount"].isna()
steps["rule 4: unparseable amounts filled from price"] = int(miss_amt.sum())
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)
neg = o["delivery_minutes"] < 0                                                 # rule 5
steps["rule 5a: negative delivery_minutes set missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
steps["rule 5b: missing delivery_minutes (completed) before fill"] = int(o.loc[comp, "delivery_minutes"].isna().sum())
o.loc[comp, "delivery_minutes"] = o[comp].groupby("category")["delivery_minutes"].transform(
    lambda s: s.fillna(s.median()))
o = o.sort_values("order_id").reset_index(drop=True)

print("\n=== CLEAN ORDERS: rule effects")
for k, v in steps.items():
    print(f"{k}: {v}")
print("rows:", len(o), "| status:", o["status"].value_counts().to_dict())
print("missing order_amount:", int(o["order_amount"].isna().sum()))
print("missing delivery (completed):", int(o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum()),
      "| missing delivery (cancelled):", int(o.loc[o.status == 'cancelled', 'delivery_minutes'].isna().sum()))
print("completed revenue:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ------------------------------------------------------------------ store
o_out = o[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
           "discount_pct", "order_amount", "delivery_minutes"]].copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod_raw.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
c_out.to_sql("customers", con, index=False)
prod_raw.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.close()
print("\nwrote data/clean/*.csv and", DB)
