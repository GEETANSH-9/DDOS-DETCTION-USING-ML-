import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset of 80K eb=ntry")

df = pd.read_csv("final_ddos_rf_model.pkl", low_memory=False)

# clean column names
df.columns = df.columns.str.strip()


df = df.drop(columns=[
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Unnamed: 0"
], errors="ignore")

# drop features unsuitable for live monitoring
drop_features = [
    "Idle Mean","Idle Std","Idle Max","Idle Min",
    "Active Mean","Active Std","Active Max","Active Min",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes",
    "Subflow Bwd Packets","Subflow Bwd Bytes"
]

df = df.drop(columns=drop_features, errors="ignore")

# clean dataset
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Dataset shape after cleaning:", df.shape)

# split features / label
X = df.drop(columns=["Label"])
y = df["Label"]

# keep numeric columns only
X = X.select_dtypes(include=[np.number])

print("Feature count:", X.shape[1])

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Training finished.")

# feature importance
importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 25 Features:")
print(feature_importance.head(25))

# save top features
top_features = feature_importance.head(20)["Feature"]

top_features.to_csv("selected_features.csv", index=False)

print("\nTop features saved to selected_features.csv")
#TO be used after removing realtime papram and adding syn ack pack  
"""Min Packet Length
Fwd Packet Length Min
Fwd Packet Length Mean
Packet Length Mean
Average Packet Size
Avg Fwd Segment Size
Fwd Packet Length Max
Total Length of Fwd Packets
Inbound
Max Packet Length
Bwd Packets/s
Destination Port
Protocol
Bwd Header Length
Init_Win_bytes_forward
Packet Length Std
Source Port
Total Backward Packets
Packet Length Variance
URG Flag Count

SYN Flag Count
ACK Flag Count
Fwd Packets/s"""