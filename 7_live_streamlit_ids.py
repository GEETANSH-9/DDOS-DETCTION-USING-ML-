import importlib.util
import threading
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DETECTOR_FILE = Path("5caputerdanalazed.py")
THREAT_ORDER = ["NORMAL", "WARMUP", "SUSPECT", "MEDIUM", "HIGH", "ATTACK"]
THREAT_COLORS = {
    "NORMAL": "green",
    "WARMUP": "dodgerblue",
    "SUSPECT": "gold",
    "MEDIUM": "orange",
    "HIGH": "darkorange",
    "ATTACK": "red",
}
EMOJI_BITS = ["✅", "🚨", "🔴", "🟠", "🟡", "⚠️", "⏳"]


def load_backend():
    spec = importlib.util.spec_from_file_location("capture_detector", DETECTOR_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def get_backend():
    return load_backend()


def strip_threat_icons(value):
    text = str(value)
    for token in EMOJI_BITS:
        text = text.replace(token, "")
    return text.strip()


def get_risk_score(threat, reason, ml_prob):
    threat = strip_threat_icons(threat)
    reason = str(reason)
    ml_prob = float(ml_prob)

    if threat == "ATTACK":
        return max(ml_prob, 0.97)
    if threat == "HIGH":
        return max(min(ml_prob, 0.96), 0.85)
    if threat == "MEDIUM":
        return max(min(ml_prob, 0.84), 0.65)
    if threat == "SUSPECT":
        return max(min(ml_prob, 0.64), 0.45)
    if threat == "WARMUP":
        return 0.20
    if "Benign bidirectional flow" in reason:
        return 0.08
    if "Low-signal one-way flow" in reason:
        return 0.12
    if "Small unconfirmed flow" in reason:
        return 0.10
    return min(ml_prob, 0.18)


def prepare_flow_rows(df):
    if df.empty:
        return df

    df = df.copy()
    df["threat_clean"] = df["threat"].apply(strip_threat_icons)
    df["risk_score"] = df.apply(
        lambda row: get_risk_score(row["threat"], row["reason"], row["ml_prob"]),
        axis=1,
    )

    for col in ["fwd_pps", "bwd_pps", "avg_size", "ml_prob", "risk_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.sort_values(["risk_score", "fwd_pkts"], ascending=[False, False]).reset_index(drop=True)


def pick_strongest_threat(values):
    order = {
        "NORMAL": 0,
        "WARMUP": 1,
        "SUSPECT": 2,
        "MEDIUM": 3,
        "HIGH": 4,
        "ATTACK": 5,
    }
    cleaned = [strip_threat_icons(value) for value in values]
    return max(cleaned, key=lambda item: order.get(item, -1))


def pick_first_real_text(values):
    for value in values:
        text = str(value)
        if text and text != "-":
            return text
    return "-"


def build_table_rows(df):
    if df.empty:
        return df

    grouped = (
        df.groupby("flow", as_index=False)
        .agg(
            fwd_pkts=("fwd_pkts", "sum"),
            bwd_pkts=("bwd_pkts", "sum"),
            fwd_pps=("fwd_pps", "sum"),
            bwd_pps=("bwd_pps", "sum"),
            avg_size=("avg_size", "mean"),
            syn=("syn", "sum"),
            ack=("ack", "sum"),
            init_win=("init_win", "max"),
            threat_clean=("threat_clean", pick_strongest_threat),
            reason=("reason", pick_first_real_text),
            risk_score=("risk_score", "max"),
            ml_prob=("ml_prob", "max"),
            rule_verdict=("rule_verdict", pick_first_real_text),
            network=("network", "last"),
        )
        .sort_values(["risk_score", "fwd_pkts"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped


def start_capture_if_needed(detector):
    if st.session_state.get("capture_started"):
        return

    detector.running = True
    sniff_thread = threading.Thread(target=detector.start_sniffing, daemon=True)
    sniff_thread.start()

    st.session_state.capture_started = True
    st.session_state.capture_started_at = time.time()
    st.session_state.capture_thread = sniff_thread


def reset_everything(detector):
    with detector.flow_lock:
        detector.flow_stats.clear()

    detector.attack_log.clear()

    if hasattr(detector, "last_logged_alert"):
        detector.last_logged_alert.clear()

    log_file = Path("attack_log.csv")
    if log_file.exists():
        log_file.unlink()


def add_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: black;
            }
            .block-container {
                max-width: 1360px;
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            .note-box {
                background: darkslategray;
                border-left: 4px solid dodgerblue;
                padding: 0.75rem 1rem;
                margin: 0.5rem 0 1.1rem 0;
                color: lightgray;
                font-size: 0.95rem;
            }
            .stat-card {
                border: 1px solid gray;
                border-radius: 8px;
                background: darkslategray;
                padding: 0.85rem 0.9rem;
                min-height: 104px;
            }
            .stat-label {
                color: darkgray;
                font-size: 0.86rem;
            }
            .stat-value {
                color: white;
                font-size: 1.55rem;
                font-weight: 700;
                margin-top: 0.25rem;
            }
            .stat-small {
                color: lightgray;
                font-size: 0.8rem;
                margin-top: 0.35rem;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid gray;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_sidebar(detector):
    with st.sidebar:
        st.header("Capture")

        if st.button("Start Capture", use_container_width=True):
            start_capture_if_needed(detector)

        if st.button("Reset State", use_container_width=True):
            reset_everything(detector)

        refresh_seconds = st.slider("Auto refresh seconds", 1, 10, 2, 1)
        min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)
        selected_threats = st.multiselect(
            "Show threat levels",
            options=THREAT_ORDER[::-1],
            default=THREAT_ORDER[::-1],
        )

    return refresh_seconds, min_risk, selected_threats


def show_header():
    st.title("DDoS live dashboard")
    st.markdown(
        """
        <div class="note-box">
            This page uses the detector code from <b>5caputerdanalazed.py</b>.
            
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_summary_cards(detector, table_df, network_state):
    started_at = st.session_state.get("capture_started_at")
    started_label = time.strftime("%H:%M:%S", time.localtime(started_at)) if started_at else "--"
    capture_status = "Running" if st.session_state.get("capture_started") else "Stopped"

    cards = [
        ("Capture Status", capture_status, f"Started at: {started_label}"),
        ("Active Flows", f"{len(table_df):,}", "IP pairs after grouping"),
        ("Network State", network_state, "From 5caputerdanalazed.py"),
        ("Logged Alerts", f"{len(detector.attack_log):,}", "Saved to attack_log.csv"),
    ]

    cols = st.columns(4)
    for col, (title, value, small_text) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{title}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-small">{small_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_threat_chart(flows_df):
    st.subheader("Current threat counts")

    if flows_df.empty:
        st.info("Threat counts will appear after capture starts.")
        return

    chart_df = (
        flows_df["threat_clean"]
        .astype(str)
        .value_counts()
        .rename_axis("Threat")
        .reset_index(name="Count")
    )
    chart_df["Threat"] = pd.Categorical(chart_df["Threat"], categories=THREAT_ORDER, ordered=True)
    chart_df = chart_df.sort_values("Threat").dropna().reset_index(drop=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Threat:N", title="Threat"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Threat:N",
                scale=alt.Scale(
                    domain=THREAT_ORDER,
                    range=[THREAT_COLORS[name] for name in THREAT_ORDER],
                ),
                legend=None,
            ),
            tooltip=["Threat", "Count"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def show_live_table(table_df):
    st.subheader("Live Flow Table")

    if table_df.empty:
        st.info("Press Start Capture and wait for a few packets to arrive.")
        return

    visible = table_df[
        [
            "flow",
            "fwd_pkts",
            "bwd_pkts",
            "fwd_pps",
            "bwd_pps",
            "avg_size",
            "syn",
            "ack",
            "init_win",
            "threat_clean",
            "reason",
            "risk_score",
            "ml_prob",
            "rule_verdict",
            "network",
        ]
    ].rename(
        columns={
            "flow": "Flow",
            "fwd_pkts": "Fwd Pkts",
            "bwd_pkts": "Bwd Pkts",
            "fwd_pps": "Fwd PPS",
            "bwd_pps": "Bwd PPS",
            "avg_size": "Avg Size",
            "syn": "SYN",
            "ack": "ACK",
            "init_win": "InitWin",
            "threat_clean": "Threat",
            "reason": "Reason",
            "risk_score": "Risk Score",
            "ml_prob": "Raw ML Prob",
            "rule_verdict": "Rule Verdict",
            "network": "Network",
        }
    )

    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Raw ML Prob": st.column_config.NumberColumn("Raw ML Prob", format="%.2f"),
            "Fwd PPS": st.column_config.NumberColumn("Fwd PPS", format="%.2f"),
            "Bwd PPS": st.column_config.NumberColumn("Bwd PPS", format="%.2f"),
            "Avg Size": st.column_config.NumberColumn("Avg Size", format="%.1f"),
        },
    )


def main():
    st.set_page_config(page_title="DDoS Live Dashboard", layout="wide")
    add_css()

    detector = get_backend()
    refresh_seconds, min_risk, selected_threats = show_sidebar(detector)
    show_header()

    raw_df, network_state = detector.get_dashboard_snapshot(log_alerts=True)
    flows_df = prepare_flow_rows(raw_df)

    if not flows_df.empty:
        flows_df = flows_df[flows_df["risk_score"] >= min_risk]
        if selected_threats:
            flows_df = flows_df[flows_df["threat_clean"].isin(selected_threats)]
        table_df = build_table_rows(flows_df)
    else:
        table_df = flows_df

    show_summary_cards(detector, table_df, network_state)
    show_threat_chart(flows_df)
    show_live_table(table_df)

    time.sleep(refresh_seconds)
    st.rerun()


if __name__ == "__main__":
    main()