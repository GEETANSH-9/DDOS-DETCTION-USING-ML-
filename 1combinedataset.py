import pandas as pd
import numpy as np

files = [
    "DrDoS_DNS.csv",
    "DrDoS_LDAP.csv",
    "DrDoS_UDP.csv",
    "DrDoS_NetBIOS.csv",
    #"real_benign_traffic.csv"
    
]

sample_size_per_file = 100000
chunksize = 50000

samples = []

for file in files:
    print("Processing:", file)

    for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False):

        # clean column names
        chunk.columns = chunk.columns.str.strip()

        # remove infinity values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # sample from this chunk
        if len(chunk) > 0:
            sample = chunk.sample(
                n=min(len(chunk), sample_size_per_file // 10),
                random_state=42
            )
            samples.append(sample)

df = pd.concat(samples, ignore_index=True)

print("\nCombined sampled dataset size:", df.shape)

print("\nClass distribution before balancing:")
print(df["Label"].value_counts())

# separate classes before  converting labels
attack_df = df[df["Label"] != "BENIGN"]
benign_df = df[df["Label"] == "BENIGN"]

print("Attack rows:", len(attack_df))
print("Benign rows:", len(benign_df))

# oversample benign
benign_sample = benign_df.sample(n=20000, replace=True, random_state=42)

# sample attacks
attack_sample = attack_df.sample(n=60000, random_state=42)

balanced_df = pd.concat([attack_sample, benign_sample])

balanced_df = balanced_df.sample(frac=1, random_state=42)

print("\nBalanced distribution BEFORE binary conversion:")
print(balanced_df["Label"].value_counts())

# Convert labels
balanced_df["Label"] = balanced_df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)
print(balanced_df["Label"].value_counts())
balanced_df.to_csv("ddos_training_dataset.csv", index=False)

print("\nDataset saved as ddos_training_dataset.csv")