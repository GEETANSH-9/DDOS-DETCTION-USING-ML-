import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,r2_score
import joblib

print("Loading dataset...")

df = pd.read_csv("ddos_training_dataset.csv", low_memory=False)

# clean column names
df.columns = df.columns.str.strip()

# remove identifier columns
df = df.drop(columns=[
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Unnamed: 0"
], errors="ignore")

# final selected features
features = [
"Min Packet Length",
"Fwd Packet Length Min",
"Fwd Packet Length Mean",
"Packet Length Mean",
"Average Packet Size",
"Avg Fwd Segment Size",
"Fwd Packet Length Max",
"Total Length of Fwd Packets",
"Inbound",
"Max Packet Length",
"Bwd Packets/s",
"Destination Port",
"Protocol",
"Bwd Header Length",
"Init_Win_bytes_forward",
"Packet Length Std",
"Source Port",
"Total Backward Packets",
"Packet Length Variance",
"URG Flag Count",
"SYN Flag Count",
"ACK Flag Count",
"Fwd Packets/s"
]

# keep only those features
X = df[features]
y = df["Label"]
X = X.select_dtypes(include=[np.number])

# clean data
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# align labels
y = y.loc[X.index]

print("Dataset shape:", X.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45,
    stratify=y
)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Training complete")
y_pred = rf.predict(X_test)
y_predt=rf.predict(X_train)
print("\nModel Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("R2 of model ",r2_score(y_train,y_predt))
print("R2 of test  ",r2_score(y_test,y_pred))


# save model
joblib.dump(rf, "final_ddos_rf_model.pkl")

print("\nModel saved as final_ddos_rf_model.pkl")

pd.Series(features).to_csv("final_features.csv", index=False)
print("Feature list saved as final_features.csv")