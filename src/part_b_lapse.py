"""Part B tasks 4-8: lapse problem, leak-free preprocessing, tuning, balancing, persistence.
Run after part_b_segments.py:  python src/part_b_lapse.py"""
import joblib
import os
os.makedirs("outputs", exist_ok=True)
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]

# ------------------------------------------------------------------ task 4: cohort, label, features
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cu = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"]).set_index("customer_id")
pr = pd.read_csv("data/clean/products.csv")
o = o.merge(pr[["product_id", "category"]], on="product_id")

before = o[o.order_ts < CUTOFF]                      # ONLY orders before the cutoff feed the features
after_completed = o[(o.order_ts >= CUTOFF) & (o.status == "completed")]   # used ONLY for the label
comp_b = before[before.status == "completed"]

feat = comp_b.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                         monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                         avg_delivery=("delivery_minutes", "mean"))
feat["recency"] = (CUTOFF - feat["last_ts"]).dt.days
cancelled_n = before[before.status == "cancelled"].groupby("customer_id").size()
final_n = before.groupby("customer_id").size()
feat["cancel_rate"] = (cancelled_n.reindex(feat.index).fillna(0) / final_n.reindex(feat.index)).astype(float)
fav = (comp_b.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat["fav_category"] = fav
feat = feat.join(cu[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat.index.isin(after_completed.customer_id.unique())).astype(int)
data = feat[NUM + CAT + ["lapsed"]]
X, y = data[NUM + CAT], data["lapsed"]

print("=== Lapse problem")
print(f"cohort: {len(data)} | lapsed: {int(y.sum())} ({y.mean()*100:.2f}%)")
print("features use only orders with order_ts <", CUTOFF.date(), "| max feature order_ts:", comp_b.order_ts.max())
assert before.order_ts.max() < CUTOFF
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train: {len(X_tr)} test: {len(X_te)} | train lapsed {int(y_tr.sum())} / not {int((y_tr==0).sum())}")

base = DummyClassifier(strategy="constant", constant=0).fit(X_tr, y_tr)
bp = base.predict(X_te)
print(f"baseline (always not lapsed): accuracy={accuracy_score(y_te, bp)*100:.1f}% "
      f"lapsed-class F1={f1_score(y_te, bp, zero_division=0):.2f}")

# ------------------------------------------------------------------ task 5: leak-free preprocessing
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

pre_demo = make_pre().fit(X_tr)                      # fitted on TRAIN rows only
Xt = pre_demo.transform(X_tr); Xe = pre_demo.transform(X_te)
print("\n=== Preprocessing")
print("scaler learned mean of frequency (train rows):", round(pre_demo.named_transformers_["num"].mean_[NUM.index("frequency")], 4),
      "| full-cohort mean of frequency:", round(X["frequency"].mean(), 4))
print("missing values in feature columns:", int(X.isna().sum().sum()), "| in encoded train/test:",
      int(np.isnan(Xt.toarray() if hasattr(Xt, 'toarray') else Xt).sum()), int(np.isnan(Xe.toarray() if hasattr(Xe, 'toarray') else Xe).sum()))
assert X.isna().sum().sum() == 0
print("encoded feature count:", Xt.shape[1])

# ------------------------------------------------------------------ task 6: tuning and evaluation
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
gs_lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
gs_rf = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_tr, y_tr)
print("\n=== Grid search")
print("LR best:", gs_lr.best_params_, "cv F1 =", round(gs_lr.best_score_, 4))
print("RF best:", gs_rf.best_params_, "cv F1 =", round(gs_rf.best_score_, 4), "| fits:", len(gs_rf.cv_results_["params"]) * 5)

def metrics(model, Xs, ys):
    p = model.predict(Xs)
    return dict(cm=confusion_matrix(ys, p), precision=precision_score(ys, p, zero_division=0),
                recall=recall_score(ys, p, zero_division=0), f1=f1_score(ys, p, zero_division=0))

print("\n=== Test-set evaluation (lapsed = positive class)")
res = {}
for name, gs in [("LogisticRegression", gs_lr), ("RandomForest", gs_rf)]:
    m = metrics(gs.best_estimator_, X_te, y_te); res[name] = m
    print(f"{name}: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")
    print("  confusion [[TN FP],[FN TP]] =", m["cm"].tolist())

sel_name = max(res, key=lambda k: res[k]["f1"])
best_model = {"LogisticRegression": gs_lr, "RandomForest": gs_rf}[sel_name].best_estimator_
print("\nselected model:", sel_name, "| test F1:", round(res[sel_name]["f1"], 3))
assert res[sel_name]["f1"] >= 0.55

# ------------------------------------------------------------------ task 7: class-balance experiment
rng_idx = np.random.RandomState(42)
lapsed_idx = np.where(y_tr.values == 1)[0]
need = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = rng_idx.choice(lapsed_idx, size=need, replace=True)
X_bal = pd.concat([X_tr, X_tr.iloc[extra]]); y_bal = pd.concat([y_tr, y_tr.iloc[extra]])
print("\n=== Balancing (train only)")
print("original train:", y_tr.value_counts().to_dict(), "| balanced train:", y_bal.value_counts().to_dict())
lr_bal = clone(gs_lr.best_estimator_).fit(X_bal, y_bal)
mb = metrics(lr_bal, X_te, y_te); mu = res["LogisticRegression"]
print(f"LR unbalanced : precision={mu['precision']:.3f} recall={mu['recall']:.3f} F1={mu['f1']:.3f} cm={mu['cm'].tolist()}")
print(f"LR oversampled: precision={mb['precision']:.3f} recall={mb['recall']:.3f} F1={mb['f1']:.3f} cm={mb['cm'].tolist()}")

# ------------------------------------------------------------------ task 8: interpret, persist, use
rf_best = gs_rf.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\n=== Top-5 RF feature importances")
print(imp.head(5).round(4).to_string())
lr_best = gs_lr.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["pre"].get_feature_names_out())
print("\nLR coefficients (standardised numerics):")
print(coef.loc[["num__recency", "num__frequency", "num__tenure_days", "num__monetary", "num__avg_delivery"]].round(3).to_string())

joblib.dump(best_model, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(best_model.predict(X_te), re.predict(X_te)) and np.allclose(best_model.predict_proba(X_te), re.predict_proba(X_te))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
proba = pd.Series(best_model.predict_proba(X_te)[:, 1], index=X_te.index)
prem = proba[proba.index.isin(seg.index[seg.is_premium])].sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": prem.index, "lapse_probability": prem.values.round(4),
                    "actually_lapsed": y_te.loc[prem.index].values})
ret.to_csv("outputs/retention_list.csv", index=False)
premium_in_test = int(proba.index.isin(seg.index[seg.is_premium]).sum())
print(f"\npremium customers in test set: {premium_in_test} | retention list rows: {len(ret)} | truly lapsed: {int(ret.actually_lapsed.sum())}")
print(ret.to_string(index=False))
