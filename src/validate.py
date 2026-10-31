"""Validation of Part A reference checkpoints.  python src/validate.py"""
import sqlite3
import pandas as pd

raw = pd.read_csv("data/raw/orders.csv")
rc = pd.read_csv("data/raw/customers.csv")
c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
comp = o[o.status == "completed"]

checks = {
    "raw order rows == 13084": len(raw) == 13084,
    "exact duplicates == 119": int(raw.duplicated().sum()) == 119,
    "pending rows == 656": int((raw.status == "pending").sum()) == 656,
    "distinct raw order_id == 12309": raw.order_id.nunique() == 12309,
    "non-numeric amounts == 120": int(pd.to_numeric(raw.order_amount, errors="coerce").isna().sum()) == 120,
    "negative delivery == 106": int((raw.delivery_minutes < 0).sum()) == 106,
    "blank delivery == 2541": int(raw.delivery_minutes.isna().sum()) == 2541,
    "blank ages == 61": int(rc.age.isna().sum()) == 61,
    "ages outside 10-100 == 6": int(((rc.age < 10) | (rc.age > 100)).sum()) == 6,
    "raw city spellings == 24": rc.city.nunique() == 24,
    "clean cities == 6": c.city.nunique() == 6,
    "clean gender counts": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "clean ages complete and in 18-60": c.age.notna().all() and c.age.between(18, 60).all(),
    "clean order rows == 12309 and unique": len(o) == 12309 and o.order_id.is_unique,
    "completed == 11110, cancelled == 1199": (len(comp), int((o.status == "cancelled").sum())) == (11110, 1199),
    "no pending rows": int((o.status == "pending").sum()) == 0,
    "no missing order_amount": int(o.order_amount.isna().sum()) == 0,
    "completed delivery complete": int(comp.delivery_minutes.isna().sum()) == 0,
    "cancelled delivery missing == 1199": int(o[o.status == "cancelled"].delivery_minutes.isna().sum()) == 1199,
    "completed revenue == 2998668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "db has 3 tables": {r[0] for r in con.execute("select name from sqlite_master where type='table'")} >= {"customers", "products", "orders"},
    "db orders count == 12309": con.execute("select count(*) from orders").fetchone()[0] == 12309,
}
failed = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
