import os
import sys
import json
import random
import requests
import streamlit as st
import numpy as np

# Configure path so we can run from anywhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
FASTAPI_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Manual Attack Simulator", layout="wide")

@st.cache_data
def load_data():
    """Loads all data necessary to sample attacks and benign traffic."""
    # 1. Load feature names
    with open(os.path.join(DATA_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    
    # 2. Load attacks
    X_attacks = np.load(os.path.join(DATA_DIR, "X_attacks.npy"))
    y_attacks = np.load(os.path.join(DATA_DIR, "y_attacks_str.npy"), allow_pickle=True).astype(str)
    
    # 3. Load benign sample
    X_benign_full = np.load(os.path.join(DATA_DIR, "X_train_benign.npy"))
    # Just take a subset of benign to save RAM in Streamlit
    benign_idx = np.random.choice(len(X_benign_full), size=10000, replace=False)
    X_benign = X_benign_full[benign_idx]
    y_benign = np.full(len(X_benign), "Benign")
    
    # Combine
    X_all = np.concatenate([X_attacks, X_benign], axis=0)
    y_all = np.concatenate([y_attacks, y_benign], axis=0)
    
    # Get unique classes
    classes = sorted(list(np.unique(y_all)))
    
    return feature_names, X_all, y_all, classes

# ── UI ─────────────────────────────────────────────────────────────
st.title("🎯 Manual Attack Simulator")
st.markdown("Select an attack type (or Benign) to inject a real logged flow into the backend.")

try:
    with st.spinner("Loading dataset into memory..."):
        feature_names, X_all, y_all, available_classes = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Move 'Benign' to the top of the list if it exists
if "Benign" in available_classes:
    available_classes.remove("Benign")
    available_classes.insert(0, "Benign")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Configure Payload")
    selected_class = st.selectbox("Select Traffic Type", available_classes)
    
    send_btn = st.button(f"Generate & Send '{selected_class}'", type="primary", use_container_width=True)

    if send_btn:
        with st.spinner(f"Sampling {selected_class}..."):
            # Filter dataset for selected class
            class_indices = np.where(y_all == selected_class)[0]
            if len(class_indices) == 0:
                st.error("No data found for this class.")
                st.stop()
                
            # Pick a random row
            idx = random.choice(class_indices)
            raw_features = X_all[idx]
            
            # Map features to a dict
            payload = {name: float(val) for name, val in zip(feature_names, raw_features)}
            
            # Additional required fields for the schema
            payload["src_ip"] = "192.168.1.100"
            payload["dst_ip"] = "10.0.0.5"
            payload["src_port"] = random.randint(1024, 65535)
            payload["dst_port"] = 80
            payload["protocol"] = 6
            payload["true_label"] = selected_class
            
            st.session_state["last_payload"] = payload
            
        with st.spinner("Sending to FastAPI..."):
            try:
                response = requests.post(FASTAPI_URL, json=payload, timeout=5)
                st.session_state["last_response_status"] = response.status_code
                st.session_state["last_response_body"] = response.text
                if response.status_code == 200:
                    st.success("Successfully sent to backend!")
                else:
                    st.warning(f"Backend returned HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the FastAPI backend running on localhost:8000?")
                st.session_state["last_response_status"] = None
                st.session_state["last_response_body"] = "Failed to connect to backend."
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("2. Network Logs")
    
    if "last_payload" in st.session_state:
        st.markdown("**Outgoing Payload (JSON):**")
        with st.expander("View Full JSON Payload", expanded=False):
            st.json(st.session_state["last_payload"])
            
        st.markdown("**Backend Response:**")
        status = st.session_state.get("last_response_status", "N/A")
        body = st.session_state.get("last_response_body", "")
        if status == 200:
            st.success(f"HTTP {status} - {body}")
        else:
            st.error(f"HTTP {status} - {body}")
    else:
        st.info("No payload sent yet. Select an attack and click the button.")
