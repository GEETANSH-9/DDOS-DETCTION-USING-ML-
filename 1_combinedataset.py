import pandas as pd
import numpy as np
import os

#  CONFIG

ATTACK_FILES = [
    "DrDoS_DNS.csv",
    "DrDoS_LDAP.csv",
    "DrDoS_UDP.csv",
    "DrDoS_NetBIOS.csv",
]

REAL_BENIGN_FILE  = "real_benign_traffic.csv"
CHUNKSIZE         = 50000
ATTACK_TARGET     = 25000
BENIGN_LAB_TARGET = 12000
BENIGN_REAL_TARGET= 12000
MAX_REAL_REPEAT   = 4       # avoid learning the same 775 wifi flows 13x over
MIN_FWD_PACKETS   = 3       # match the runtime detector's minimum flow size


#  STEP 1: Sample from lab CSVs

attack_samples     = []
benign_lab_samples = []

for file in ATTACK_FILES:
    if not os.path.exists(file):
        print(f"WARNING: {file} not found, skipping")
        continue
    print(f"Processing: {file}")
    for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)
        if "Total Fwd Packets" in chunk.columns:
            chunk = chunk[chunk["Total Fwd Packets"] >= MIN_FWD_PACKETS]
        if len(chunk) == 0:
            continue
        attacks = chunk[chunk["Label"] != "BENIGN"]
        benigns = chunk[chunk["Label"] == "BENIGN"]
        if len(attacks) > 0:
            attack_samples.append(attacks.sample(
                n=min(len(attacks), CHUNKSIZE // 5), random_state=42))
        if len(benigns) > 0:
            benign_lab_samples.append(benigns.sample(
                n=min(len(benigns), CHUNKSIZE // 10), random_state=42))

attack_df     = pd.concat(attack_samples, ignore_index=True)
benign_lab_df = pd.concat(benign_lab_samples, ignore_index=True) if benign_lab_samples else pd.DataFrame()

print(f"\nRaw attack rows   : {len(attack_df)}")
print(f"Raw lab benign    : {len(benign_lab_df)}")


#  STEP 2: Load real benign

if os.path.exists(REAL_BENIGN_FILE):
    real_benign_df = pd.read_csv(REAL_BENIGN_FILE)
    real_benign_df.columns = real_benign_df.columns.str.strip()
    if "Fwd Packets/s" in real_benign_df.columns:
        real_benign_df = real_benign_df[real_benign_df["Fwd Packets/s"] > 0]
    real_benign_df["Label"] = 0
    print(f"Real benign rows  : {len(real_benign_df)}")
else:
    real_benign_df = pd.DataFrame()
    print("WARNING: real_benign_traffic.csv not found — using lab benign only")


#  STEP 3: Build balanced dataset

attack_df["Label"] = 1
attack_df["CaptureSource"] = "attack_lab"

if len(benign_lab_df) > 0:
    benign_lab_df["Label"] = 0
    benign_lab_df["CaptureSource"] = "benign_lab"
    lab_sample = benign_lab_df.sample(
        n=min(len(benign_lab_df), BENIGN_LAB_TARGET),
        replace=len(benign_lab_df) < BENIGN_LAB_TARGET,
        random_state=42)
else:
    lab_sample = pd.DataFrame()

if len(real_benign_df) > 0:
    real_benign_df["CaptureSource"] = "benign_real"
    real_target = min(BENIGN_REAL_TARGET, len(real_benign_df) * MAX_REAL_REPEAT)
    real_sample = real_benign_df.sample(
        n=real_target,
        replace=real_target > len(real_benign_df),
        random_state=42)
else:
    real_sample = pd.DataFrame()

benign_pool = pd.concat(
    [x for x in [lab_sample, real_sample] if len(x) > 0],
    ignore_index=True)

dedupe_cols = [c for c in benign_pool.columns if c not in {"CaptureSource"}]
if len(benign_pool) > 0:
    benign_pool = benign_pool.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

effective_attack_target = min(len(attack_df), ATTACK_TARGET, len(benign_pool))
attack_sample = attack_df.sample(n=effective_attack_target, random_state=42)

balanced_df = pd.concat([attack_sample, benign_pool], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42)

print(f"\n── Final Dataset ")
print(f"Attack          : {len(attack_sample)}")
print(f"Lab benign      : {len(lab_sample)}")
print(f"Real benign     : {len(real_sample)} (oversampled from {len(real_benign_df)})")
print(f"Total           : {len(balanced_df)}")
print(balanced_df["Label"].value_counts())

balanced_df.to_csv("ddos_training_dataset.csv", index=False)
print("\nSaved → ddos_training_dataset.csv")
