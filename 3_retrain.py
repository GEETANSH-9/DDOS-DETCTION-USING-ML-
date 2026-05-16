import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

FEATURES = [
    "Min Packet Length",        "Fwd Packet Length Min",
    "Fwd Packet Length Mean",   "Packet Length Mean",
    "Average Packet Size",      "Avg Fwd Segment Size",
    "Fwd Packet Length Max",    "Total Length of Fwd Packets",
    "Inbound",                  "Max Packet Length",
    "Bwd Packets/s",            "Destination Port",
    "Protocol",                 "Bwd Header Length",
    "Init_Win_bytes_forward",   "Packet Length Std",
    "Source Port",              "Total Backward Packets",
    "Packet Length Variance",   "URG Flag Count",
    "SYN Flag Count",           "ACK Flag Count",
    "Fwd Packets/s",
]

# Load
print("Loading dataset...")
df = pd.read_csv("ddos_training_dataset.csv", low_memory=False)
df.columns = df.columns.str.strip()
source_col = "CaptureSource" if "CaptureSource" in df.columns else None
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Unnamed: 0"]
df = df.drop(columns=drop_cols, errors="ignore")
print(f"Raw shape    : {df.shape}")
print(df["Label"].value_counts())

# Keep the offline training rows compatible with what the live sniffer can build.
before = len(df)
if "Total Fwd Packets" in df.columns:
    df = df[df["Total Fwd Packets"] >= 3]
print(f"\nAfter flow-size filter: {len(df)}  (dropped {before - len(df)})")
print(df["Label"].value_counts())

#  Prepare X / y 
X = df[FEATURES].select_dtypes(include=[np.number]).copy()
y = df["Label"]
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]
sources = df.loc[X.index, source_col] if source_col else pd.Series("unknown", index=X.index)

sample_weight_map = {
    "attack_lab": 1.0,
    "benign_lab": 1.5,
    "benign_real": 4.0,
    "unknown": 1.0,
}
sample_weights = sources.map(sample_weight_map).fillna(1.0)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

#  Model 1: Regularised RF
print("\n Regularised Random Forest ")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,           # was None — this alone fixes most overfit
    min_samples_leaf=10,    # was 1
    min_samples_split=20,
    max_features="sqrt",
    max_samples=0.7,
    class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train, sample_weight=w_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
train_rf  = rf.score(X_train, y_train)
test_rf   = accuracy_score(y_test, y_pred_rf)
print(f"Train acc : {train_rf:.4f}  |  Test acc : {test_rf:.4f}")
print(f"Gap       : {train_rf - test_rf:.4f}  (healthy < 0.02)")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

#  Model 2: XGBoost 
print("\n── XGBoost ")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,          # L1 regularisation
    reg_lambda=1.0,         # L2 regularisation
    scale_pos_weight=max((w_train[y_train == 0].sum() / max(w_train[y_train == 1].sum(), 1e-6)), 1.0),
    eval_metric="logloss",
    random_state=42, n_jobs=-1
)
xgb.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
train_xgb  = xgb.score(X_train, y_train)
test_xgb   = accuracy_score(y_test, y_pred_xgb)
print(f"\nTrain acc : {train_xgb:.4f}  |  Test acc : {test_xgb:.4f}")
print(f"Gap       : {train_xgb - test_xgb:.4f}  (healthy < 0.02)")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# Cross Validation 
print("\n── 5-Fold Cross Validation (XGBoost) ──")
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV AUC : {cv_scores.round(4)}")
print(f"Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#  Save 
print("\n── Summary ─────────────────────────────")
print(f"RF  gap: {train_rf  - test_rf :.4f}  AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"XGB gap: {train_xgb - test_xgb:.4f}  AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

joblib.dump(rf,  "final_ddos_rf_model.pkl")
joblib.dump(xgb, "final_ddos_xgb_model.pkl")
pd.Series(FEATURES).to_csv("final_features.csv", index=False)

print("\nSaved → final_ddos_rf_model.pkl")
print("Saved → final_ddos_xgb_model.pkl  (use this in capture script)")
print("Saved → final_features.csv")
