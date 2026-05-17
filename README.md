# Real-Time DDoS Detection and Classification using Machine Learning

A lightweight, real-time Network Intrusion Detection System (NIDS) that captures 
live network traffic, extracts flow-level features, and classifies traffic as 
normal or malicious using an ensemble ML approach — without requiring heavy infrastructure.

## Results

| Model | Train Accuracy | Test Accuracy | ROC-AUC | CV Mean AUC |
|---|---|---|---|---|
| Random Forest (Regularised) | 99.79% | 99.9% | 1.000 | — |
| XGBoost | 99.97% | 99.9% | 1.000 | 0.9999 ± 0.0002 |

Evaluated on CIC-DDoS2019 dataset (DNS, UDP, LDAP, NetBIOS attack types).  
Features reduced from 80+ → 23 real-time-computable features via RF importance ranking.

## Dashboard Screenshot

<img width="1429" height="794" alt="Screenshot 2026-04-16 at 12 45 13 PM" src="https://github.com/user-attachments/assets/fedce3a0-2b49-42c8-a0ad-b922edb06ca5" />

<img width="1429" height="794" alt="Screenshot 2026-04-16 at 12 45 13 PM" src="https://github.com/user-attachments/assets/b8b16bb9-c976-4fb7-b8ae-321f92119d6a" />

## How It Works

The system runs as a 5-stage pipeline:

**Stage 1 — Dataset Preparation** (`1_prepare_dataset.py`)  
Combines multiple CIC-DDoS2019 CSV files (DrDoS_DNS, LDAP, UDP, NetBIOS), 
removes duplicates, balances attack/benign traffic, and saves a clean training CSV.

**Stage 2 — Feature Selection** (`2_feature_selection.py`)  
Trains an initial Random Forest on 80+ features and ranks them by importance. 
Drops non-real-time features (IAT-based, bulk-rate metrics). Outputs top 23 features.

**Stage 3 — Model Training** (`3_train_model.py`)  
Retrains Random Forest (300 estimators) on selected features with stratified 
train/test split. Saves the final model as `final_ddos_rf_model.pkl`.

**Stage 4 — Live Capture & Analysis** (`4_capture_analyse.py`)  
Uses Scapy to sniff live packets, aggregates per source IP in a 15-second sliding 
window, extracts the 23 features in real time, and runs ML + rule-based classification.

**Threat tiers:** NORMAL → WARMUP → SUSPECT → MEDIUM → HIGH → ATTACK

**Stage 5 — Streamlit Dashboard** (`5_dashboard.py`)  
Live web dashboard showing per-flow threat levels, risk scores, forward/backward 
PPS, SYN/ACK counts, and detection reasons. Logs confirmed attacks to CSV.

## Installation

```bash
git clone https://github.com/GEETANSH-9/DDOS-DETCTION-USING-ML-.git
cd DDOS-DETCTION-USING-ML-
pip install -r requirements.txt
```

## Usage

Run each stage in order:

```bash
# 1. Prepare dataset (requires CIC-DDoS2019 CSV files in same folder)
python 1_prepare_dataset.py

# 2. Select features
python 2_feature_selection.py

# 3. Train model
python 3_train_model.py

# 4. Start live capture (run in background terminal, requires sudo/admin)
sudo python 4_capture_analyse.py

# 5. Launch dashboard (in a second terminal)
streamlit run 5_dashboard.py
```

## Dataset

[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) — Canadian Institute 
for Cybersecurity, University of New Brunswick.  
Attack types used: DrDoS_DNS, DrDoS_LDAP, DrDoS_UDP, DrDoS_NetBIOS.

## Tech Stack

Python · Scapy · Scikit-learn · XGBoost · Pandas · NumPy · Streamlit · Altair · Joblib

## Author

Geetansh Pasrija — [LinkedIn](https://linkedin.com/in/geetansh-pasrija-bb1731259) · 
[LeetCode](https://leetcode.com/u/geetansh_18/)
