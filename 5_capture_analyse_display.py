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
FLOW_WARMUP_SECONDS = 2.0
BENIGN_PORTS = {53, 80, 123, 443}
SMALL_FLOW_MAX_PACKETS = 12
SMALL_FLOW_MAX_PPS = 80
ONE_WAY_REVIEW_PACKETS = 20
ONE_WAY_REVIEW_PPS = 120

console = Console()
running = True

# LOAD MODEL AND FEATURES
model = joblib.load("final_ddos_xgb_model.pkl")
features = pd.read_csv("final_features.csv").iloc[:, 0].tolist()
feature_importance = model.feature_importances_

attack_log = []


#  BIDIRECTIONAL FLOW STORAGE
#  Key: (src_ip, dst_ip, src_port, dst_port, protocol)
#  We normalise the key so the reverse direction
#  maps to the SAME flow entry.

flow_stats = {}
flow_lock  = threading.Lock()   # sniff thread + display thread share this dict


def make_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """
    Always return the same key regardless of which direction
    the packet travels. We sort the two endpoints so that
    (A→B) and (B→A) produce the same key.
    """
    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 <= ep2:
        return (ep1[0], ep2[0], ep1[1], ep2[1], proto)
    else:
        return (ep2[0], ep1[0], ep2[1], ep1[1], proto)


def new_flow():
    return {
        # ── forward direction (first seen src → dst) ──
        "fwd_sizes":      [],
        "fwd_timestamps": [],
        "fwd_header_len": [],   # TCP/UDP header lengths per fwd packet
        "fwd_syn":        0,
        "fwd_ack":        0,
        "fwd_urg":        0,
        "init_win_fwd":   -1,   # TCP window on very first fwd SYN; -1 = not seen yet

        # ── backward direction (reply: dst → src) ──
        "bwd_sizes":      [],
        "bwd_timestamps": [],
        "bwd_header_len": [],

        # ── shared ──
        "src_ip":   None,
        "dst_ip":   None,
        "src_port": 0,
        "dst_port": 0,
        "protocol": 0,
        "udp":      0,
        "inbound":  0,          # 1 if any packet came INTO our host, else 0
    }



#  PACKET ANALYSIS  (bidirectional)

def analyze_packet(packet):

    if not packet.haslayer(IP):
        return

    src_ip  = packet[IP].src
    dst_ip  = packet[IP].dst
    proto   = 0
    src_port = 0
    dst_port = 0
    is_tcp   = False
    is_udp   = False

    if packet.haslayer(TCP):
        proto    = 6
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        is_tcp   = True
    elif packet.haslayer(UDP):
        proto    = 17
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        is_udp   = True
    else:
        # non-TCP/UDP; still track with port 0
        pass

    key = make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)

    now     = time.time()
    pkt_len = len(packet)

    with flow_lock:

        if key not in flow_stats:
            flow_stats[key] = new_flow()
            # first packet defines forward direction
            flow_stats[key]["src_ip"]   = src_ip
            flow_stats[key]["dst_ip"]   = dst_ip
            flow_stats[key]["src_port"] = src_port
            flow_stats[key]["dst_port"] = dst_port
            flow_stats[key]["protocol"] = proto

        fs = flow_stats[key]

        # decide direction: forward = same src as when flow was created
        is_forward = (src_ip == fs["src_ip"] and src_port == fs["src_port"])

        if is_forward:
            fs["fwd_sizes"].append(pkt_len)
            fs["fwd_timestamps"].append(now)

            if is_tcp:
                hdr_len = packet[TCP].dataofs * 4   # TCP data offset → bytes
                fs["fwd_header_len"].append(hdr_len)

                flags = packet[TCP].flags
                if flags & 0x02:
                    fs["fwd_syn"] += 1
                    # capture TCP window on first SYN only
                    if fs["init_win_fwd"] == -1:
                        fs["init_win_fwd"] = packet[TCP].window
                if flags & 0x10:
                    fs["fwd_ack"] += 1
                if flags & 0x20:
                    fs["fwd_urg"] += 1

            elif is_udp:
                fs["fwd_header_len"].append(8)      # UDP header is always 8 bytes
                fs["udp"] += 1

        else:   # backward / reply direction
            fs["bwd_sizes"].append(pkt_len)
            fs["bwd_timestamps"].append(now)
            fs["inbound"] = 1                       # we received a reply → inbound

            if is_tcp:
                hdr_len = packet[TCP].dataofs * 4
                fs["bwd_header_len"].append(hdr_len)
            elif is_udp:
                fs["bwd_header_len"].append(8)

        # ── sliding-window cleanup (both directions) ──
        for direction in [("fwd_timestamps", "fwd_sizes", "fwd_header_len"),
                          ("bwd_timestamps", "bwd_sizes", "bwd_header_len")]:
            ts_key, sz_key, hdr_key = direction
            while fs[ts_key] and now - fs[ts_key][0] > WINDOW:
                fs[ts_key].pop(0)
                fs[sz_key].pop(0)
                if fs[hdr_key]:
                    fs[hdr_key].pop(0)



#  FEATURE EXTRACTION  

def build_features(fs):

    fwd_sizes = np.array(fs["fwd_sizes"])
    bwd_sizes = np.array(fs["bwd_sizes"])

    # need at least MIN_PACKETS in the forward direction
    if len(fwd_sizes) < MIN_PACKETS:
        return None

    all_sizes = np.concatenate([fwd_sizes, bwd_sizes]) if len(bwd_sizes) else fwd_sizes

    # durations
    fwd_times = np.array(fs["fwd_timestamps"])
    bwd_times = np.array(fs["bwd_timestamps"])

    fwd_duration = fwd_times[-1] - fwd_times[0] if len(fwd_times) > 1 else 1
    if fwd_duration <= 0:
        fwd_duration = 1

    bwd_duration = bwd_times[-1] - bwd_times[0] if len(bwd_times) > 1 else 1
    if bwd_duration <= 0:
        bwd_duration = 1

    f = {}

    # ── packet length features (all packets) ──
    f["Min Packet Length"]      = np.min(all_sizes)
    f["Max Packet Length"]      = np.max(all_sizes)
    f["Packet Length Mean"]     = np.mean(all_sizes)
    f["Packet Length Std"]      = np.std(all_sizes)
    f["Packet Length Variance"] = np.var(all_sizes)
    f["Average Packet Size"]    = np.mean(all_sizes)

    # ── forward-only features ──
    f["Fwd Packet Length Min"]      = np.min(fwd_sizes)
    f["Fwd Packet Length Mean"]     = np.mean(fwd_sizes)
    f["Fwd Packet Length Max"]      = np.max(fwd_sizes)
    f["Total Length of Fwd Packets"]= np.sum(fwd_sizes)
    f["Avg Fwd Segment Size"]       = np.mean(fwd_sizes)
    f["Fwd Packets/s"]              = len(fwd_sizes) / fwd_duration

    # ── backward features  (NOW REAL) ──
    f["Total Backward Packets"] = len(bwd_sizes)
    f["Bwd Packets/s"]          = len(bwd_sizes) / bwd_duration if len(bwd_sizes) > 0 else 0.0

    # Bwd Header Length: sum of actual captured header lengths, else default 20
    if fs["bwd_header_len"]:
        f["Bwd Header Length"] = float(np.mean(fs["bwd_header_len"]))
    else:
        f["Bwd Header Length"] = 20.0   # nothing seen yet; keep original default

    # ── Init window (NOW REAL from TCP SYN) ──
    f["Init_Win_bytes_forward"] = fs["init_win_fwd"] if fs["init_win_fwd"] != -1 else 0

    # ── Inbound flag (NOW REAL) ──
    f["Inbound"] = fs["inbound"]

    # ── ports / protocol ──
    f["Destination Port"] = fs["dst_port"]
    f["Source Port"]      = fs["src_port"]
    f["Protocol"]         = fs["protocol"]

    # ── flag counts (forward direction) ──
    f["SYN Flag Count"] = fs["fwd_syn"]
    f["ACK Flag Count"] = fs["fwd_ack"]
    f["URG Flag Count"] = fs["fwd_urg"]

    return f


def benign_confidence(fs, feat, fwd_duration):
    score = 0

    if len(fs["bwd_sizes"]) >= 2:
        score += 1
    if fs["fwd_ack"] >= fs["fwd_syn"] and fs["fwd_ack"] > 0:
        score += 1
    if feat["Init_Win_bytes_forward"] >= 1024:
        score += 1
    if feat["Destination Port"] in BENIGN_PORTS:
        score += 1
    if feat["Fwd Packets/s"] < 120:
        score += 1
    if fwd_duration >= FLOW_WARMUP_SECONDS:
        score += 1

    return score


def is_small_unconfirmed_flow(fs, feat, fwd_count, bwd_count, fwd_duration):
    if bwd_count > 0:
        return False
    if fwd_count > SMALL_FLOW_MAX_PACKETS:
        return False
    if feat["Fwd Packets/s"] > SMALL_FLOW_MAX_PPS:
        return False
    if fs["fwd_syn"] > 2:
        return False
    if fwd_duration >= FLOW_WARMUP_SECONDS * 2:
        return False
    return True


def is_low_signal_one_way_flow(fs, feat, fwd_count, bwd_count):
    if bwd_count > 0:
        return False
    if fs["fwd_syn"] > 1:
        return False
    if fs["fwd_ack"] > 0:
        return False
    if fwd_count >= ONE_WAY_REVIEW_PACKETS:
        return False
    if feat["Fwd Packets/s"] >= ONE_WAY_REVIEW_PPS:
        return False
    return True



#  EXPLAIN ML  (unchanged)

def explain_prediction(feat):

    df     = pd.DataFrame([feat])
    contrib = []

    for i, f in enumerate(features):
        val   = df.iloc[0][f]
        score = val * feature_importance[i]
        contrib.append((f, score))

    contrib = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)
    return ", ".join([c[0] for c in contrib[:3]])



#  DISTRIBUTED DETECTION
#  Now iterates over flow_stats instead of ip_stats

def detect_distributed_attack():

    total_pps    = 0
    high_pps_ips = 0
    seen_srcs    = set()

    with flow_lock:
        flows = list(flow_stats.values())

    for fs in flows:
        fwd_ts = fs["fwd_timestamps"]
        if len(fwd_ts) < 2:
            continue

        duration = fwd_ts[-1] - fwd_ts[0]
        if duration <= 0:
            continue

        pps = len(fwd_ts) / duration
        total_pps += pps
        seen_srcs.add(fs["src_ip"])

        if pps > 200:
            high_pps_ips += 1

    if high_pps_ips > 10 and total_pps > 5000:
        return "🚨 Distributed DDoS"

    if high_pps_ips > 5 and total_pps > 2000:
        return "⚠️ Possible DDoS"

    return "Normal"



#  DASHBOARD  

def display_dashboard():

    while running:

        network_state = detect_distributed_attack()

        table = Table(title="Real-Time ML DDoS IDS  [bidirectional]")

        table.add_column("Flow (src→dst)",  style="cyan")
        table.add_column("Fwd Pkts")
        table.add_column("Bwd Pkts")
        table.add_column("Fwd PPS")
        table.add_column("Bwd PPS")
        table.add_column("AvgSize")
        table.add_column("SYN")
        table.add_column("InitWin")
        table.add_column("Threat")
        table.add_column("Reason")
        table.add_column("ML Prob")
        table.add_column("Rule Verdict")
        table.add_column("Network")

        with flow_lock:
            flows_snapshot = list(flow_stats.items())

        for key, fs in flows_snapshot:

            feat = build_features(fs)
            if feat is None:
                continue

            fwd_sizes = np.array(fs["fwd_sizes"])
            fwd_ts    = np.array(fs["fwd_timestamps"])

            fwd_duration = fwd_ts[-1] - fwd_ts[0] if len(fwd_ts) > 1 else 1
            if fwd_duration <= 0:
                fwd_duration = 1

            fwd_count = len(fwd_sizes)
            bwd_count = len(fs["bwd_sizes"])
            fwd_pps   = fwd_count / fwd_duration
            bwd_pps   = feat["Bwd Packets/s"]
            avg       = feat["Average Packet Size"]
            syn       = fs["fwd_syn"]
            ack       = fs["fwd_ack"]
            init_win  = feat["Init_Win_bytes_forward"]

            syn_ratio = syn / max(fwd_count, 1)
            syn_rate  = syn / fwd_duration
            flow_age  = fwd_duration
            mature_flow = flow_age >= FLOW_WARMUP_SECONDS or bwd_count >= 2

            #  RULE-BASED CHECKS 
            rule_triggered = False
            rule_verdict   = "Normal"

            if syn > 3000 and syn_ratio > 0.7:
                rule_verdict   = "SYN Flood"
                rule_triggered = True

            elif syn_rate > 200:
                rule_verdict   = "SYN Rate"
                rule_triggered = True

            elif syn > ack * 3 and syn > 1000:
                rule_verdict   = "SYN Imbalance"
                rule_triggered = True

            elif fwd_pps > 300 and syn > 1000:
                rule_verdict   = "High PPS"
                rule_triggered = True

            elif syn_ratio > 0.5 or fwd_pps > 200:
                rule_verdict   = "Suspicious"
                rule_triggered = True

            # ── ML INFERENCE ──
            df_feat  = pd.DataFrame([feat])[features]
            ml_prob  = model.predict_proba(df_feat)[0][1]
            benign_score = benign_confidence(fs, feat, fwd_duration)
            small_unconfirmed = is_small_unconfirmed_flow(
                fs, feat, fwd_count, bwd_count, fwd_duration
            )
            low_signal_one_way = is_low_signal_one_way_flow(
                fs, feat, fwd_count, bwd_count
            )

            # ── FINAL DECISION ──
            if small_unconfirmed and not rule_triggered:
                threat = "✅ NORMAL"
                reason = "Small unconfirmed flow"

            elif low_signal_one_way and not rule_triggered:
                if ml_prob > 0.97:
                    threat = "🟡 SUSPECT"
                    reason = "One-way flow, waiting for stronger signal"
                else:
                    threat = "✅ NORMAL"
                    reason = "Low-signal one-way flow"

            elif not mature_flow and not rule_triggered:
                threat = "⏳ WARMUP"
                reason = "Collecting more packets"

            elif rule_triggered:

                if rule_verdict == "Suspicious":
                    if ml_prob > ATTACK_THRESHOLD and benign_score < 4:
                        threat = "🚨 ATTACK"
                        reason = f"Rule + ML ({ml_prob:.2f})"
                    else:
                        threat = "⚠️ SUSPECT"
                        reason = "Rule: Suspicious"
                else:
                    threat = "🚨 ATTACK"
                    reason = f"Rule: {rule_verdict}"

                attack_log.append({
                    "time":        time.strftime("%H:%M:%S"),
                    "src_ip":      fs["src_ip"],
                    "dst_ip":      fs["dst_ip"],
                    "probability": ml_prob,
                    "reason":      reason
                })

            else:

                if ml_prob > 0.97 and benign_score < 2 and fwd_count >= 10:
                    threat = "🚨 ATTACK"
                elif ml_prob > 0.92 and benign_score < 3 and fwd_count >= 8:
                    threat = "🔴 HIGH"
                elif ml_prob > 0.82 and not small_unconfirmed and not low_signal_one_way:
                    threat = "🟠 MEDIUM"
                elif ml_prob > 0.65 and not small_unconfirmed and not low_signal_one_way:
                    threat = "🟡 SUSPECT"
                else:
                    threat = "✅ NORMAL"

                reason = "-"

                if ml_prob > 0.92 and benign_score < 3 and fwd_count >= 8:
                    reason = explain_prediction(feat)
                    attack_log.append({
                        "time":        time.strftime("%H:%M:%S"),
                        "src_ip":      fs["src_ip"],
                        "dst_ip":      fs["dst_ip"],
                        "probability": ml_prob,
                        "reason":      reason
                    })
                elif ml_prob > 0.7 and benign_score >= 4:
                    threat = "✅ NORMAL"
                    reason = "Benign bidirectional flow"
                elif low_signal_one_way:
                    threat = "✅ NORMAL"
                    reason = "Low-signal one-way flow"

            flow_label = f"{fs['src_ip']}→{fs['dst_ip']}"

            table.add_row(
                flow_label,
                str(fwd_count),
                str(bwd_count),
                f"{fwd_pps:.1f}",
                f"{bwd_pps:.1f}",
                f"{avg:.1f}",
                str(syn),
                str(init_win),
                threat,
                reason,
                f"{ml_prob:.2f}",
                rule_verdict,
                network_state
            )

        console.clear()
        console.print(table)
        time.sleep(2)



#  SNIFFER  

def start_sniffing():
    sniff(prn=analyze_packet, store=False, filter="ip")



#  SAVE REPORT  

def save_report():
    if len(attack_log) > 0:
        pd.DataFrame(attack_log).to_csv("attack_log.csv", index=False)
        console.print("📁 Attack log saved")


#  MAIN  (unchanged)

if __name__ == "__main__":

    console.print("[bold green]Starting ML IDS (bidirectional)... Press Ctrl+C to stop[/bold green]")

    sniff_thread = threading.Thread(target=start_sniffing, daemon=True)
    sniff_thread.start()

    try:
        display_dashboard()

    except KeyboardInterrupt:
        running = False
        console.print("\nStopping detector")
        save_report()
