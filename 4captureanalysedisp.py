from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time
import threading
import numpy as np
import pandas as pd
import joblib
from rich.console import Console
from rich.table import Table

WINDOW = 15
MIN_PACKETS = 3
ATTACK_THRESHOLD = 0.8

console = Console()
running = True

# LOAD MODEL AND FEATURES
model = joblib.load("final_ddos_rf_model.pkl")
features = pd.read_csv("final_features.csv").iloc[:, 0].tolist()
feature_importance = model.feature_importances_

attack_log = []

# PER-IP STORAGE
ip_stats = defaultdict(lambda: {
    "sizes": [],
    "timestamps": [],
    "syn": 0,
    "ack": 0,
    "urg": 0,
    "udp": 0,
    "src_port": 0,
    "dst_port": 0,
    "protocol": 0
})


# PACKET ANALYSIS 
def analyze_packet(packet):

    if not packet.haslayer(IP):
        return

    src = packet[IP].src
    stats = ip_stats[src]

    now = time.time()

    stats["sizes"].append(len(packet))
    stats["timestamps"].append(now)

    while stats["timestamps"] and now - stats["timestamps"][0] > WINDOW:
        stats["timestamps"].pop(0)
        stats["sizes"].pop(0)

    if packet.haslayer(TCP):
        stats["protocol"] = 6
        stats["src_port"] = packet[TCP].sport
        stats["dst_port"] = packet[TCP].dport

        flags = packet[TCP].flags
        if flags & 0x02:
            stats["syn"] += 1
        if flags & 0x10:
            stats["ack"] += 1
        if flags & 0x20:
            stats["urg"] += 1

    elif packet.haslayer(UDP):
        stats["protocol"] = 17
        stats["src_port"] = packet[UDP].sport
        stats["dst_port"] = packet[UDP].dport
        stats["udp"] += 1


#  FEATURE EXTRACTION 
def build_features(stats):

    if len(stats["sizes"]) < MIN_PACKETS:
        return None

    sizes = np.array(stats["sizes"])
    times = np.array(stats["timestamps"])

    duration = times[-1] - times[0]
    if duration <= 0:
        duration = 1

    f = {}

    f["Min Packet Length"] = np.min(sizes)
    f["Fwd Packet Length Min"] = np.min(sizes)
    f["Fwd Packet Length Mean"] = np.mean(sizes)
    f["Packet Length Mean"] = np.mean(sizes)
    f["Average Packet Size"] = np.mean(sizes)
    f["Avg Fwd Segment Size"] = np.mean(sizes)
    f["Fwd Packet Length Max"] = np.max(sizes)
    f["Total Length of Fwd Packets"] = np.sum(sizes)

    f["Inbound"] = 1
    f["Max Packet Length"] = np.max(sizes)

    f["Bwd Packets/s"] = len(sizes) / duration
    f["Fwd Packets/s"] = len(sizes) / duration

    f["Destination Port"] = stats["dst_port"]
    f["Source Port"] = stats["src_port"]

    f["Protocol"] = stats["protocol"]

    f["Bwd Header Length"] = 20
    f["Init_Win_bytes_forward"] = 0

    f["Packet Length Std"] = np.std(sizes)
    f["Packet Length Variance"] = np.var(sizes)

    f["Total Backward Packets"] = len(sizes)

    f["URG Flag Count"] = stats["urg"]
    f["SYN Flag Count"] = stats["syn"]
    f["ACK Flag Count"] = stats["ack"]

    return f


# EXPLAIN ML 
def explain_prediction(feat):

    df = pd.DataFrame([feat])
    contrib = []

    for i, f in enumerate(features):
        val = df.iloc[0][f]
        score = val * feature_importance[i]
        contrib.append((f, score))

    contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)
    return ", ".join([c[0] for c in contrib[:3]])


#  DISTRIBUTED DETECTION 
def detect_distributed_attack():

    active_ips = 0
    total_pps = 0
    high_pps_ips = 0

    for stats in ip_stats.values():

        if len(stats["timestamps"]) < 2:
            continue

        duration = stats["timestamps"][-1] - stats["timestamps"][0]
        if duration <= 0:
            continue

        pps = len(stats["timestamps"]) / duration
        total_pps += pps
        active_ips += 1

        if pps > 200:
            high_pps_ips += 1

    if high_pps_ips > 10 and total_pps > 5000:
        return "🚨 Distributed DDoS"

    if high_pps_ips > 5 and total_pps > 2000:
        return "⚠️ Possible DDoS"

    return "Normal"


#  DASHBOARD RICH TABLE 
def display_dashboard():

    while running:

        network_state = detect_distributed_attack()

        table = Table(title=" Real-Time ML DDoS IDS")

        table.add_column("IP", style="cyan")
        table.add_column("Packets")
        table.add_column("PPS")
        table.add_column("AvgSize")
        table.add_column("SYN")
        table.add_column("UDP")
        table.add_column("Threat")
        table.add_column("Reason")
        table.add_column("ML Prob")
        table.add_column("Rule Verdict")
        table.add_column("Network")

        for ip, stats in list(ip_stats.items()):

            feat = build_features(stats)
            if feat is None:
                continue

            sizes = np.array(stats["sizes"])
            duration = stats["timestamps"][-1] - stats["timestamps"][0]
            if duration <= 0:
                duration = 1

            packet_count = len(sizes)
            pps = packet_count / duration
            avg = np.mean(sizes)

            syn = stats["syn"]
            ack = stats["ack"]

            syn_ratio = syn / max(packet_count, 1)
            syn_rate = syn / duration

            # RULES MANUAL
            rule_triggered = False
            rule_verdict = "Normal"

            if syn > 3000 and syn_ratio > 0.7:
                rule_verdict = "SYN Flood"
                rule_triggered = True

            elif syn_rate > 200:
                rule_verdict = "SYN Rate"
                rule_triggered = True

            elif syn > ack * 3 and syn > 1000:
                rule_verdict = "SYN Imbalance"
                rule_triggered = True

            elif pps > 300 and syn > 1000:
                rule_verdict = "High PPS"
                rule_triggered = True

            elif syn_ratio > 0.5 or pps > 200:
                rule_verdict = "Suspicious"
                rule_triggered = True

            
            df = pd.DataFrame([feat])[features]
            ml_prob = model.predict_proba(df)[0][1]

            #  FINAL DECISION 
            if rule_triggered:

                if rule_verdict == "Suspicious":
                    if ml_prob > ATTACK_THRESHOLD:
                        threat = "🚨 ATTACK"
                        reason = f"Rule + ML ({ml_prob:.2f})"
                    else:
                        threat = "⚠️ SUSPECT"
                        reason = "Rule: Suspicious"
                else:
                    threat = "🚨 ATTACK"
                    reason = f"Rule: {rule_verdict}"

                attack_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "ip": ip,
                    "probability": ml_prob,
                    "reason": reason
                })

            else:

                if ml_prob > 0.9:
                    threat = " 🔴HIGH"
                elif ml_prob > 0.75:
                    threat = " MEDIUM"
                elif ml_prob > 0.6:
                    threat = " SUSPECT"
                else:
                    threat = "✅ NORMAL"

                reason = "-"

                if ml_prob > ATTACK_THRESHOLD:
                    reason = explain_prediction(feat)

                    attack_log.append({
                        "time": time.strftime("%H:%M:%S"),
                        "ip": ip,
                        "probability": ml_prob
                    })

            table.add_row(
                ip,
                str(packet_count),
                f"{pps:.1f}",
                f"{avg:.1f}",
                str(syn),
                str(stats["udp"]),
                threat,
                reason,
                f"{ml_prob:.2f}",
                rule_verdict,
                network_state
            )

        console.clear()
        console.print(table)
        time.sleep(2)


# SNIFFER 
def start_sniffing():
    sniff(prn=analyze_packet, store=False, filter="ip")


# SAVE REPORT 
def save_report():
    if len(attack_log) > 0:
        pd.DataFrame(attack_log).to_csv("attack_log.csv", index=False)
        console.print("📁 Attack log saved")


#  MAIN 
if __name__ == "__main__":

    console.print("[bold green]Starting ML IDS... Press Ctrl+C to stop[/bold green]")

    sniff_thread = threading.Thread(target=start_sniffing, daemon=True)
    sniff_thread.start()

    try:
        display_dashboard()

    except KeyboardInterrupt:
        running = False
        console.print("\nStopping detector")
        save_report()