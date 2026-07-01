import os
import json
import random
import requests
import streamlit as st
import numpy as np
from datetime import datetime

# ── Path Setup ─────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR      = os.path.join(_PROJECT_ROOT, "data", "processed")
FASTAPI_URL   = "http://localhost:8000/predict"

# ── Attack metadata: icon + severity ──────────────────────────────
ATTACK_META = {
    "Botnet_ARES":       {"icon": "🤖", "severity": "CRITICAL", "color": "#ff2244"},
    "DDoS_LOIT":         {"icon": "🌊", "severity": "CRITICAL", "color": "#ff2244"},
    "DoS_GoldenEye":     {"icon": "👁️",  "severity": "HIGH",     "color": "#ff8800"},
    "DoS_Hulk":          {"icon": "💪", "severity": "HIGH",     "color": "#ff8800"},
    "DoS_Slowhttptest":  {"icon": "🐢", "severity": "MEDIUM",   "color": "#ffcc00"},
    "DoS_Slowloris":     {"icon": "🕷️",  "severity": "MEDIUM",   "color": "#ffcc00"},
    "FTP-Patator":       {"icon": "🔑", "severity": "HIGH",     "color": "#ff8800"},
    "Port_Scan":         {"icon": "📡", "severity": "MEDIUM",   "color": "#ffcc00"},
    "SSH-Patator":       {"icon": "🔐", "severity": "HIGH",     "color": "#ff8800"},
    "Web_Brute_Force":   {"icon": "🪓", "severity": "HIGH",     "color": "#ff8800"},
    "Web_XSS":           {"icon": "💉", "severity": "MEDIUM",   "color": "#ffcc00"},
    "Benign":            {"icon": "✅", "severity": "SAFE",     "color": "#00ff88"},
}

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThreatSentinel — Attack Injector",
    page_icon="☠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS: dark terminal theme ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

/* ── Background ── */
html, body, [data-testid="stApp"] {
    background-color: #080c10 !important;
    color: #00ff88 !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top left, #0a1a0f 0%, #050810 60%, #000 100%);
}

/* ── All text ── */
*, p, span, label, div {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff88;
}

/* ── Header / hero ── */
.hero-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.6rem;
    font-weight: 700;
    color: #00ff88;
    text-shadow: 0 0 18px #00ff88, 0 0 40px #00cc66;
    letter-spacing: 4px;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #4dffaa;
    font-size: 0.85rem;
    letter-spacing: 2px;
    opacity: 0.75;
    margin-bottom: 1.5rem;
}
.divider {
    border: none;
    border-top: 1px solid #00ff8844;
    margin: 1rem 0;
    box-shadow: 0 0 6px #00ff8833;
}

/* ── Panels ── */
.panel {
    background: #0d1f15;
    border: 1px solid #00ff8833;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 0 20px #00ff8811, inset 0 0 30px #00ff8808;
}
.panel-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #00ff88;
    text-shadow: 0 0 8px #00ff88;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #00ff8833;
    padding-bottom: 0.4rem;
}

/* ── Attack badge ── */
.badge-critical { color: #ff2244; text-shadow: 0 0 8px #ff2244; }
.badge-high     { color: #ff8800; text-shadow: 0 0 8px #ff8800; }
.badge-medium   { color: #ffcc00; text-shadow: 0 0 8px #ffcc00; }
.badge-safe     { color: #00ff88; text-shadow: 0 0 8px #00ff88; }

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div {
    background: #0a1a0f !important;
    border: 1px solid #00ff8844 !important;
    border-radius: 4px !important;
    color: #00ff88 !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #003311, #005522) !important;
    color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    border-radius: 4px !important;
    box-shadow: 0 0 12px #00ff8833 !important;
    transition: all 0.2s ease !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #005522, #00aa44) !important;
    box-shadow: 0 0 24px #00ff8866 !important;
}

/* ── Log output box ── */
.log-box {
    background: #020a04;
    border: 1px solid #00ff8822;
    border-radius: 4px;
    padding: 1rem;
    font-size: 0.78rem;
    line-height: 1.6;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    color: #00ff88;
    box-shadow: inset 0 0 20px #00ff8808;
}

/* ── Status indicators ── */
.status-ok   { color: #00ff88; text-shadow: 0 0 8px #00ff88; }
.status-err  { color: #ff2244; text-shadow: 0 0 8px #ff2244; }
.status-warn { color: #ffcc00; text-shadow: 0 0 8px #ffcc00; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    background: #0a1a0f;
    border: 1px solid #00ff8822;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.metric-val {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.3rem;
    color: #00ff88;
    text-shadow: 0 0 10px #00ff88;
}
.metric-lbl {
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    color: #4dffaa;
    opacity: 0.7;
}

/* ── Spinner / alerts ── */
[data-testid="stSpinner"] { color: #00ff88 !important; }
[data-testid="stAlert"] {
    background: #0a1a0f !important;
    border: 1px solid #00ff8833 !important;
    color: #00ff88 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #040c08; }
::-webkit-scrollbar-thumb { background: #00ff8844; border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #060e08 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0a1a0f !important;
    border: 1px solid #00ff8822 !important;
    border-radius: 4px !important;
}

/* ── Top scan line animation ── */
@keyframes scanline {
    0%   { transform: translateY(-100%); opacity: 0.03; }
    100% { transform: translateY(100vh); opacity: 0.03; }
}
.scanline {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(transparent, #00ff88, transparent);
    animation: scanline 4s linear infinite;
    pointer-events: none;
    z-index: 9999;
}
</style>

<div class="scanline"></div>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(os.path.join(DATA_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    X_attacks = np.load(os.path.join(DATA_DIR, "X_attacks.npy"))
    y_attacks  = np.load(os.path.join(DATA_DIR, "y_attacks_str.npy"), allow_pickle=True).astype(str)
    X_benign_full = np.load(os.path.join(DATA_DIR, "X_train_benign.npy"))
    benign_idx    = np.random.choice(len(X_benign_full), size=10000, replace=False)
    X_benign = X_benign_full[benign_idx]
    y_benign = np.full(len(X_benign), "Benign")
    X_all = np.concatenate([X_attacks, X_benign], axis=0)
    y_all = np.concatenate([y_attacks,  y_benign], axis=0)
    # per-class counts
    classes, counts = np.unique(y_all, return_counts=True)
    class_counts = dict(zip(classes.tolist(), counts.tolist()))
    return feature_names, X_all, y_all, sorted(classes.tolist()), class_counts


# ── Hero header ───────────────────────────────────────────────────
st.markdown('<div class="hero-title">☠ THREATSENTINEL</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">[ ATTACK INJECTION CONSOLE ]  //  CICIDS2017 REPLAY ENGINE</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────
try:
    with st.spinner("[ LOADING THREAT DATABASE ... ]"):
        feature_names, X_all, y_all, available_classes, class_counts = load_data()
except Exception as e:
    st.markdown(f'<div class="status-err">⚠ FATAL: Failed to load dataset — {e}</div>', unsafe_allow_html=True)
    st.stop()

total_flows = len(X_all)
attack_flows = int((y_all != "Benign").sum())

# Reorder: Benign first
if "Benign" in available_classes:
    available_classes.remove("Benign")
    available_classes.insert(0, "Benign")

# ── Top metric row ─────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-val">{total_flows:,}</div>
    <div class="metric-lbl">TOTAL FLOWS LOADED</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{attack_flows:,}</div>
    <div class="metric-lbl">ATTACK FLOWS</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{len(available_classes) - 1}</div>
    <div class="metric-lbl">ATTACK CLASSES</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{len(feature_names)}</div>
    <div class="metric-lbl">FEATURE DIMENSIONS</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    st.markdown('<div class="panel-title">▶ TARGET SELECTION</div>', unsafe_allow_html=True)

    selected_class = st.selectbox(
        "ATTACK VECTOR",
        available_classes,
        format_func=lambda x: f"{ATTACK_META.get(x, {}).get('icon','❓')}  {x}"
    )

    # Info card for selected class
    meta     = ATTACK_META.get(selected_class, {"icon": "❓", "severity": "UNKNOWN", "color": "#aaaaaa"})
    sev      = meta["severity"].lower()
    sev_cls  = f"badge-{sev}" if sev in ("critical","high","medium","safe") else "status-warn"
    cnt      = class_counts.get(selected_class, 0)

    st.markdown(f"""
    <div class="panel" style="margin-top:0.8rem; border-color:{meta['color']}44;">
        <div style="font-size:2rem; text-align:center;">{meta['icon']}</div>
        <div style="text-align:center; margin-top:0.3rem;">
            <span style="font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:2px;
                         color:{meta['color']}; text-shadow:0 0 8px {meta['color']};">
                {selected_class}
            </span>
        </div>
        <div style="text-align:center; margin-top:0.4rem;">
            <span class="{sev_cls}" style="font-size:0.7rem; letter-spacing:2px;">
                ◈ SEVERITY: {meta['severity']}
            </span>
        </div>
        <div style="text-align:center; margin-top:0.3rem; opacity:0.6; font-size:0.7rem;">
            {cnt:,} samples in database
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fire_btn = st.button("⚡  INJECT PAYLOAD", use_container_width=True)


with right_col:
    st.markdown('<div class="panel-title">▶ TRANSMISSION LOG</div>', unsafe_allow_html=True)

    # Session log
    if "log_lines" not in st.session_state:
        st.session_state["log_lines"] = [
            f"[{datetime.now().strftime('%H:%M:%S')}] SYSTEM ONLINE — threat injection console ready",
            f"[{datetime.now().strftime('%H:%M:%S')}] {total_flows:,} flows loaded across {len(available_classes)} classes",
            f"[{datetime.now().strftime('%H:%M:%S')}] TARGET: {FASTAPI_URL}",
            "[─────────────────────────────────────────────────]",
            "[ AWAITING INJECTION COMMAND ... ]",
        ]

    if fire_btn:
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        class_indices = np.where(y_all == selected_class)[0]
        if len(class_indices) == 0:
            st.session_state["log_lines"].append(f"[{ts}] ✗ ERROR — no samples found for class '{selected_class}'")
        else:
            idx          = random.choice(class_indices.tolist())
            raw_features = X_all[idx]
            payload      = {name: float(val) for name, val in zip(feature_names, raw_features)}
            payload["src_ip"]    = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
            payload["dst_ip"]    = f"10.0.{random.randint(0,10)}.{random.randint(1,254)}"
            payload["src_port"]  = random.randint(1024, 65535)
            payload["dst_port"]  = random.choice([80, 443, 21, 22, 8080])
            payload["protocol"]  = 6
            payload["true_label"]= selected_class

            st.session_state["last_payload"] = payload
            meta = ATTACK_META.get(selected_class, {"icon": "❓", "severity": "?"})

            st.session_state["log_lines"].append("[─────────────────────────────────────────────────]")
            st.session_state["log_lines"].append(f"[{ts}] {meta['icon']}  FIRING {selected_class.upper()} PAYLOAD")
            st.session_state["log_lines"].append(f"[{ts}] ► SRC  {payload['src_ip']}:{payload['src_port']}")
            st.session_state["log_lines"].append(f"[{ts}] ► DST  {payload['dst_ip']}:{payload['dst_port']}")
            st.session_state["log_lines"].append(f"[{ts}] ► SAMPLE IDX  #{idx}  |  {len(feature_names)} FEATURES")
            st.session_state["log_lines"].append(f"[{ts}] ► SENDING POST → {FASTAPI_URL}")

            try:
                response = requests.post(FASTAPI_URL, json=payload, timeout=5)
                st.session_state["last_response_status"] = response.status_code
                st.session_state["last_response_body"]   = response.text
                if response.status_code == 200:
                    st.session_state["log_lines"].append(f"[{ts}] ✔ HTTP 200 — PAYLOAD ACCEPTED")
                    st.session_state["log_lines"].append(f"[{ts}] ◈ RESPONSE: {response.text[:200]}")
                else:
                    st.session_state["log_lines"].append(f"[{ts}] ⚠ HTTP {response.status_code} — {response.text[:120]}")
            except requests.exceptions.ConnectionError:
                st.session_state["last_response_status"] = None
                st.session_state["last_response_body"]   = "CONNECTION REFUSED"
                st.session_state["log_lines"].append(f"[{ts}] ✗ CONNECTION REFUSED — backend offline at {FASTAPI_URL}")
                st.session_state["log_lines"].append(f"[{ts}]   → Start backend: uvicorn backend.main:app --reload")
            except Exception as e:
                st.session_state["log_lines"].append(f"[{ts}] ✗ ERROR — {str(e)}")

    # Render log
    log_text = "\n".join(st.session_state["log_lines"][-60:])
    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

    # Payload expander
    if "last_payload" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">▶ RAW PAYLOAD (JSON)</div>', unsafe_allow_html=True)
        with st.expander("[ EXPAND TO VIEW FULL 115-FEATURE VECTOR ]", expanded=False):
            st.json(st.session_state["last_payload"])
