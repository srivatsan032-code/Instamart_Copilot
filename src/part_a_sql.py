"""Part A: run the seven SQL queries and the pandas cross-check / EDA.  python src/part_a_sql.py"""
import re, sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
Q = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
R = {k: pd.read_sql_query(v, con) for k, v in Q.items()}

print("Q1 customers:", len(R["Q1"]))
q2 = R["Q2"]
print("Q2 rows:", len(q2), "| sum:", round(q2.revenue.sum(), 2), "| largest:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(R["Q3"]))
print("Q4 customers with no completed order:", len(R["Q4"]))
print("Q5 customers above average:", len(R["Q5"]), "| average:", R["Q5"].avg_total.iloc[0])
q6 = R["Q6"]
print("Q6 top per city:\n", q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative:", q7.cumulative_revenue.iloc[-1])

# ---------------- pandas cross-check
o = pd.read_sql_query("SELECT * FROM orders WHERE status='completed'", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
m = o.merge(c, on="customer_id").merge(p, on="product_id")
pv = pd.pivot_table(m, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("\npivot grand total:", round(pv.loc["All", "All"], 2), "| equals SQL total:",
      bool(np.isclose(pv.loc["All", "All"], q2.revenue.sum())))
print(pv.round(2).to_string())
agg = m.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                            n=("order_amount", "count")).round(2)
print("\nper-city aggregate:\n", agg.to_string())

# ---------------- EDA on completed order_amount
a = o["order_amount"]
q1_, q3_ = a.quantile(.25), a.quantile(.75)
iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
print(f"\nmean={a.mean():.2f} median={a.median():.2f} skew={a.skew():.3f} Q1={q1_:.2f} Q3={q3_:.2f} "
      f"lower_fence={lo:.2f} upper_fence={hi:.2f} below={int((a<lo).sum())} above={int((a>hi).sum())}")
corr = o[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
print("largest |corr| pair:", tri.abs().idxmax(), round(tri[tri.abs().idxmax()], 3))
