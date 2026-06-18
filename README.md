# 🛡️ Real-Time Incident Tracking & Autonomous Threat Remediation(Detailed Note)

> A two-stage deep learning pipeline for autonomous network intrusion detection and remediation using PyTorch Autoencoders and Deep Q-Networks.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Research & Study Log](#research--study-log)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Datasets](#datasets)
- [Project Milestones](#project-milestones)
- [Research References](#research-references)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## Project Overview

Traditional cyber defense relies on human analysts who take hours to respond to alerts. By that time, automated ransomware has already encrypted entire databases. This project builds an **autonomous, intelligent security system** that:

- **Detects** anomalous network traffic in real time using an unsupervised Reconstruction Autoencoder
- **Responds** autonomously using a Deep Q-Network (DQN) agent that selects the optimal containment action in milliseconds — without human intervention
- **Displays** live alerts and incident history on a React dashboard via WebSocket

### The Two-Stage Pipeline

```
Replay Simulator (CICIDS2017 CSV rows sent as live HTTP POST)
       │  POST /predict  (raw 115-feature flow + metadata)
       ▼
┌─────────────────────┐
│  FastAPI Backend    │   Apply scaler.pkl → Autoencoder forward pass
│  /predict endpoint  │   Reconstruction error = ‖x − x̂‖²
└─────────┬───────────┘
          │  error > threshold?
          ├── NO  → Log benign, discard
          ▼  YES
┌─────────────────────┐
│  Stage 1            │   Anomaly confirmed
│  Autoencoder        │   Trained ONLY on benign traffic
│  (Anomaly Detector) │   → High error = attack detected
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 2            │   Selects from 5 actions:
│  DQN Agent          │   Block IP / Revoke creds /
│  (Auto-Remediation) │   Isolate server / Kill process / Monitor
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  Dummy Action Executor                              │
│  Simulated network state dict (IP → status)         │
│  No real firewall/router — demo-safe                │
└─────────┬───────────────────────────────────────────┘
          │
          ▼
   PostgreSQL + WebSocket broadcast
          │
          ▼
   React Dashboard (live alert feed + network status map)
```

---

## Research & Study Log

All research, learning, and design decisions made during this project, logged by date.

---

### 📅 11 June 2026 — Project Kickoff & Problem Understanding

**Topics covered:**
- Understood the full project brief: real-time anomaly detection + autonomous remediation
- Studied what ransomware actually does at the network level
- Understood why human response time (hours/days) is the core bottleneck
- Learned the difference between signature-based IDS (rule-based, misses zero-days) and anomaly-based IDS (detects unknown attacks)

**Key decisions made:**
- Use unsupervised Autoencoder for Stage 1 (no labels needed, catches zero-day attacks)
- Use Deep Q-Network for Stage 2 (learns optimal response policy through simulation)
- Primary dataset: CICIDS2017 | Secondary dataset: UNSW-NB15

**Concepts understood:**
- Reconstruction error as anomaly score: `L = ‖x − x̂‖²`
- Why training only on BENIGN rows makes the model generalize to unseen attacks

---

### 📅 12 June 2026 — Network Security Domain Knowledge

**Topics covered:**
- What network flow logs contain (src/dst IP, ports, protocol, packet counts, byte counts, duration, TCP flags)
- CICIDS2017 dataset: 78 features per flow, 14 attack categories, 2.8M rows
- Attack types studied in depth:

| Category | Attacks |
|---|---|
| Denial of Service | DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS SlowHTTPTest, DDoS |
| Reconnaissance | PortScan, FTP-Patator, SSH-Patator |
| Infiltration | Infiltration, Heartbleed, Bot |
| Web Attacks | Brute Force, XSS, SQL Injection |

**Key insight:** CICIDS2017 has severe class imbalance (~80% BENIGN). The autoencoder approach is ideal because it trains only on the abundant normal class — rarity of attacks becomes irrelevant.

**Detection fingerprints learned:**
- DoS Hulk: enormous forward packet count, very short duration
- PortScan: one source IP hitting many destination ports, lots of SYN/RST flags
- Bot: periodic outbound connections at fixed intervals (heartbeat pattern)
- Heartbleed: port 443, small request → unusually large response

---

### 📅 13 June 2026 — Full Stack Application Architecture Design

**Topics covered:**
- Designed the complete 4-layer application architecture
- Understood the offline training vs online production separation
- Defined the 4 artefact files that bridge training and production

**Architecture layers defined:**

```
Layer 1 — Data Layer       : Pandas, NumPy, Scikit-learn, PostgreSQL
Layer 2 — ML Layer         : PyTorch + CUDA, Autoencoder, DQN, Gymnasium
Layer 3 — Backend Layer    : FastAPI, WebSockets, SQLAlchemy, Uvicorn
Layer 4 — Frontend Layer   : React.js, Recharts, WebSocket client
```

**4 artefact files produced by training:**
- `autoencoder.pt` — trained autoencoder weights
- `dqn_agent.pt` — trained DQN agent weights
- `scaler.pkl` — fitted MinMaxScaler parameters
- `threshold.json` — anomaly detection threshold value

**Runtime inference pipeline (one network flow):**
```
Receive flow → Preprocess (scaler) → Autoencoder forward pass
→ Compute reconstruction error → Compare to threshold
→ [Normal] Log & discard
→ [Anomaly] DQN selects action → Execute → Store in PostgreSQL → Push via WebSocket
```

---

### 📅 14 June 2026 — Python for Machine Learning

**Topics covered:**

**NumPy:**
- Arrays and shapes: `data.shape` → `(2273097, 78)`
- Array math without loops: `np.mean(data, axis=0)`, `np.std(data, axis=0)`
- Min-max normalization: `(data - mn) / (mx - mn)`
- Boolean masking: `data[labels == "BENIGN"]`
- Indexing: rows, columns, slices
- Key formula: `np.sum((x - x_hat)**2)` = reconstruction error

**Pandas:**
- Loading CSVs: `pd.read_csv("cicids2017.csv")`
- Stripping column names: `df.columns.str.strip()` — critical for CICIDS2017
- Cleaning: `replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()`
- Exploring: `df['Label'].value_counts()`, `df.describe()`
- Splitting features from labels: `X = df.drop(columns=['Label']).values`

**Matplotlib:**
- Class distribution bar chart
- Reconstruction error histogram (normal vs attacks)
- Training loss curve (train loss vs val loss per epoch)

**Complete data prep pipeline:**
```python
df = pd.read_csv("cicids2017.csv")
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
X = df.drop(columns=['Label']).values.astype('float32')
y = df['Label'].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_normal = X_scaled[y == 'BENIGN']
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
```

---

### 📅 15 June 2026 — Machine Learning Fundamentals

**Topics covered:**

**1. Supervised vs Unsupervised Learning**
- Supervised: model learns from labeled data (your DQN uses reward signals)
- Unsupervised: model finds patterns without labels (your Autoencoder)
- Why unsupervised for Stage 1: catches zero-day attacks the model has never seen

**2. Loss Functions**
- MSE (Mean Squared Error): `loss = ((x - x_hat)**2).mean()` — used in autoencoder
- Binary Cross-Entropy: used for supervised classification
- Reward signal: `+10` threat neutralized, `-3` false alarm, `-5` service disrupted — used in DQN

**3. Gradient Descent**
- Weights updated every mini-batch: `weight = weight - lr * gradient`
- In PyTorch: `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`
- Learning rate: too high = unstable, too low = slow. Typical: `0.001`
- Mini-batch gradient descent: update every N samples (e.g. `batch_size=256`)

**4. Train / Validation / Test Split**
- Training (70%): model learns from this
- Validation (15%): tune hyperparameters, monitor overfitting
- Test (15%): touch ONCE at the very end — final performance measurement
- **Critical rule:** fit scaler only on training data, `.transform()` val and test

**5. Overfitting vs Underfitting**
- Underfitting: both train and val loss high → model too simple
- Overfitting: train loss low, val loss rising → model memorizing
- Fixes: Dropout `nn.Dropout(p=0.2)`, early stopping, L2 regularization `weight_decay=1e-5`

**6. Evaluation Metrics (why accuracy is useless here)**
- Precision = TP / (TP + FP) → how many alarms are real
- Recall = TP / (TP + FN) → how many real attacks are caught (most critical for security)
- F1 Score = 2 × (P × R) / (P + R) → primary comparison metric
- ROC-AUC → threshold-independent performance measure
- **Target metrics:** Recall > 0.90, Precision > 0.85, F1 > 0.87, ROC-AUC > 0.95

---

### 📅 15 June 2026 — Dataset Analysis

**CICIDS2017 files downloaded from Kaggle:**

| File | Size | Type |
|---|---|---|
| monday_benign.csv | 386,208 KB | Benign traffic (training) |
| tuesday_benign.csv | 310,456 KB | Benign traffic (training) |
| wednesday_benign.csv | 310,869 KB | Benign traffic (training) |
| thursday_benign.csv | 105,350 KB | Benign traffic (training) |
| friday_benign.csv | 285,306 KB | Benign traffic (training) |
| dos_hulk.csv | 268,098 KB | Attack — DoS Hulk |
| ddos_loit.csv | 88,972 KB | Attack — DDoS LOIT |
| portscan.csv | 115,457 KB | Attack — PortScan |
| botnet_ares.csv | 3,877 KB | Attack — Botnet ARES |
| dos_golden_eye.csv | 7,846 KB | Attack — DoS GoldenEye |
| dos_slowhttptest.csv | 5,137 KB | Attack — DoS SlowHTTPTest |
| dos_slowloris.csv | 4,151 KB | Attack — DoS Slowloris |
| ftp_patator.csv | 7,290 KB | Attack — FTP-Patator |
| heartbleed.csv | 12 KB | Attack — Heartbleed (very few rows) |
| ssh_patator_new.csv | 4,802 KB | Attack — SSH-Patator |
| web_brute_force.csv | 1,874 KB | Attack — Web Brute Force |
| web_sql_injection.csv | 21 KB | Attack — SQL Injection |
| web_xss.csv | 913 KB | Attack — XSS |

**UNSW-NB15 files downloaded from Kaggle:**

| File | Size | Notes |
|---|---|---|
| NUSW-NB15_features.csv | 4 KB | Feature descriptions — read first |
| UNSW-NB15_1.csv | 165,020 KB | Data (no header row) |
| UNSW-NB15_2.csv | 161,349 KB | Data (no header row) |
| UNSW-NB15_3.csv | 150,965 KB | Data (no header row) |
| UNSW-NB15_4.csv | 95,302 KB | Data (no header row) |

**Dataset strategy decided:**
- **Primary:** CICIDS2017 — full training and evaluation pipeline
- **Secondary:** UNSW-NB15 — cross-dataset generalization test (train on CICIDS, test on UNSW without retraining)
- Strategy A (separate models) chosen over combined training — appropriate for beginner implementation

**Loading UNSW-NB15 correctly:**
```python
# UNSW-NB15 has NO header row — must assign column names manually
features = pd.read_csv("NUSW-NB15_features.csv", encoding='latin-1')
df = pd.concat([pd.read_csv(f"UNSW-NB15_{i}.csv", header=None)
                for i in range(1, 5)], ignore_index=True)
df.columns = features['Name'].tolist()
# Labels: 0 = normal, 1 = attack (different from CICIDS text labels)
```

---

### 📅 15 June 2026 — Project Charter & Planning

**Project Charter filled with:**
- Project Name: Real-Time Incident Tracking & Autonomous Threat Remediation
- Estimated Costs: $0 (open-source tools only)
- Problem: Human response time hours/days vs millisecond attack speed
- Business Case: Average breach cost $4.88M (IBM 2024); 207 days avg detection time
- Goals: Recall > 90%, Precision > 85%, F1 > 0.87, Response < 50ms

**Official milestone schedule (mentor-approved):**

| Milestone | Start | Finish |
|---|---|---|
| AI Opportunity Identification & Feasibility | 11/06/2026 | 13/06/2026 |
| AI Project Planning & Architecture Design | 13/06/2026 | 15/06/2026 |
| Data Readiness & Engineering | 15/06/2026 | 21/06/2026 |
| AI Model Prototyping | 21/06/2026 | 29/06/2026 |
| Model Development & Optimization | 29/06/2026 | 08/07/2026 |
| AI Solution Integration & Deployment | 08/07/2026 | 17/07/2026 |
| AI Validation, Governance & Release Readiness | 17/07/2026 | 22/07/2026 |
| Production Monitoring & Continuous Improvement | 22/07/2026 | 24/07/2026 |
| Project Closure & Handover | 24/07/2026 | 25/07/2026 |


### 📅 16 June 2026 — Data Preprocessing Pipeline for CICIDS2017

**Topics covered:**
- Why preprocessing is mandatory before training the autoencoder
- All 7 preprocessing steps specific to CICIDS2017
- Identified known data quality issues in the CICIDS2017 dataset

**7 preprocessing steps defined and understood:**

#### Step 1 — Load and combine all CSV files
- Benign files (5 files) combined separately from attack files (13 files)
- Benign data = autoencoder training set only
- Attack data = evaluation only, never seen by autoencoder during training
- Memory tip: use `low_memory=False` in `pd.read_csv()` for large files like dos_hulk.csv (268MB)

#### Step 2 — Fix column names ⚠️ CICIDS-specific gotcha
- All column names have invisible leading/trailing spaces baked into the CSV
- `df['Label']` silently fails with KeyError because actual column name is `' Label'`
- Fix: `df.columns = df.columns.str.strip()` — run immediately after loading
- Also drop any unnamed index columns: `df.loc[:, ~df.columns.str.contains('^Unnamed')]`

#### Step 3 — Remove infinity values and NaN
- CICIDS2017 contains `inf` and `-inf` from CICFlowMeter division-by-zero bugs
- Affected columns: `Flow Bytes/s`, `Flow Packets/s` (divide by zero when duration = 0)
- `NaN` appears in std deviation features when a flow has only one packet
- A single `inf` in a batch makes the entire loss `nan` — training dies immediately
- Fix: `df.replace([np.inf, -np.inf], np.nan).dropna()`
- Typically removes ~2,000–5,000 rows out of 1.3M — acceptable loss

#### Step 4 — Remove duplicate rows
- CICIDS2017 contains ~6% exact duplicate rows
- Duplicates bias what the model thinks is "normal" (some patterns over-represented)
- Fix: `df.drop_duplicates()` — run AFTER step 3, not before
- Order matters: fix inf first, then deduplicate

#### Step 5 — Fix data types
- PyTorch requires `float32` — raw CICIDS data loads as `float64`
- Using `float64` doubles GPU memory usage for no accuracy benefit
- Fix: `df.drop(columns=['Label']).astype('float32')`
- Verify no non-numeric columns remain in features (should only be Label)

#### Step 6 — Normalize features to 0–1 range ⚠️ Most critical step
- CICIDS features span wildly different ranges: Flow Duration (0–119,999,999) vs Header Length (0–43,772)
- Without normalization, large-value features dominate; small features contribute nothing
- Use MinMaxScaler — fit ONLY on training data, transform on val/test
- **Data leakage rule:** never call `fit_transform()` on val or test data
- Save fitted scaler: `pickle.dump(scaler, open("models/scaler.pkl", "wb"))`

#### Step 7 — Split and save as .npy arrays
- Final arrays saved to disk so preprocessing never needs to repeat
- For autoencoder: input = target (X_train is both input and output label)
- Sanity checks before saving:
  - `np.isnan(X_train).any()` → must be `False`
  - `np.isinf(X_train).any()` → must be `False`
  - `X_train.min()` → `0.0`, `X_train.max()` → `1.0`

**Complete preprocessing script (`training/preprocess.py`):**
```python
"""
training/preprocess.py
Run once before training. Produces:
X_train.npy, X_val.npy, X_attacks.npy, y_attacks.npy, scaler.pkl
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing  import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_DIR  = "data/cicids2017/"
OUT_DIR   = "data/processed/"
MODEL_DIR = "models/"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BENIGN_FILES = ["monday_benign.csv","tuesday_benign.csv",
                "wednesday_benign.csv","thursday_benign.csv","friday_benign.csv"]
ATTACK_FILES = ["botnet_ares.csv","ddos_loit.csv","dos_golden_eye.csv",
                "dos_hulk.csv","dos_slowhttptest.csv","dos_slowloris.csv",
                "ftp_patator.csv","heartbleed.csv","portscan.csv",
                "ssh_patator_new.csv","web_brute_force.csv",
                "web_sql_injection.csv","web_xss.csv"]

print("[1/7] Loading CSV files...")
df_b = pd.concat([pd.read_csv(DATA_DIR+f) for f in BENIGN_FILES],  ignore_index=True)
df_a = pd.concat([pd.read_csv(DATA_DIR+f) for f in ATTACK_FILES], ignore_index=True)

print("[2/7] Fixing column names...")
df_b.columns = df_b.columns.str.strip()
df_a.columns = df_a.columns.str.strip()
df_b = df_b.loc[:, ~df_b.columns.str.contains('^Unnamed')]
df_a = df_a.loc[:, ~df_a.columns.str.contains('^Unnamed')]

print("[3/7] Removing inf and NaN...")
df_b = df_b.replace([np.inf, -np.inf], np.nan).dropna()
df_a = df_a.replace([np.inf, -np.inf], np.nan).dropna()

print("[4/7] Removing duplicates...")
df_b = df_b.drop_duplicates()
df_a = df_a.drop_duplicates()

print("[5/7] Separating features and converting to float32...")
X_b = df_b.drop(columns=['Label']).astype('float32')
X_a = df_a.drop(columns=['Label']).astype('float32')
y_a = df_a['Label'].values

print("[6/7] Normalizing with MinMaxScaler...")
X_train, X_val = train_test_split(X_b, test_size=0.2, random_state=42)
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train).astype('float32')
X_val   = scaler.transform(X_val).astype('float32')
X_atk   = scaler.transform(X_a).astype('float32')
with open(MODEL_DIR+"scaler.pkl","wb") as f: pickle.dump(scaler, f)

print("[7/7] Saving arrays...")
np.save(OUT_DIR+"X_train.npy",   X_train)
np.save(OUT_DIR+"X_val.npy",     X_val)
np.save(OUT_DIR+"X_attacks.npy", X_atk)
np.save(OUT_DIR+"y_attacks.npy", y_a)

print(f"\n✓ Done!")
print(f"  X_train:   {X_train.shape}")   # (~971K, 78)
print(f"  X_val:     {X_val.shape}")     # (~243K, 78)
print(f"  X_attacks: {X_atk.shape}")     # (~1.4M, 78)
print(f"  NaN check: {np.isnan(X_train).any()}")  # False
print(f"  Inf check: {np.isinf(X_train).any()}")  # False
print(f"  Range:     {X_train.min():.3f} – {X_train.max():.3f}")  # 0.0 – 1.0
```

---

### 📅 16 June 2026 — Dataset Sufficiency Analysis

**Question answered: Is 1.81 GB enough to train the autoencoder and DQN?**

**Answer: Yes — more than enough.**

**Key findings:**

| Metric | Value |
|---|---|
| Total dataset size | 1.81 GB raw CSV |
| Estimated total rows | ~2.8 million flows |
| Benign rows (autoencoder training) | ~1.3 million |
| Features per row | 78 |
| Average size in published research | ~1.0 GB |
| Minimum viable for autoencoder | ~50,000 benign rows |
| You have | ~1.3 million benign rows (26× minimum) |

**Model-specific analysis:**

- **Autoencoder:** Extremely data-efficient. Trains only on ~1.3M benign rows. Even 50K would produce a working model. No concern whatsoever.
- **DQN agent:** Does NOT train on the dataset directly. Trains in a simulated Gymnasium environment. Dataset only informs the simulation design. Dataset size is irrelevant for DQN training.

**The one real concern — class imbalance within attacks:**
- DoS Hulk: ~231,000 rows ✅ excellent
- PortScan: ~159,000 rows ✅ excellent
- DDoS: ~42,000 rows ✅ good
- Web XSS: ~2,000 rows ⚠️ limited
- Heartbleed: ~11 rows ❌ statistically meaningless

> **Important note for project report:** Heartbleed performance metrics will not be statistically valid due to only ~11 available rows. This is a documented limitation of CICIDS2017 itself, not of the model. Must be mentioned explicitly in the evaluation section.

**Training time estimates (with CUDA GPU):**
- One epoch over 1M rows at batch_size=256: ~2–5 minutes
- 50 epochs total: ~1.5–4 hours
- Without GPU (CPU only): multiply by 10–20×

**Memory usage after preprocessing:**
- 1.3M rows × 78 features × 4 bytes (float32) = ~390 MB RAM
- Fits comfortably in 8 GB RAM

---

### 📅 16 June 2026 — Backpropagation in Autoencoders

**Question answered: Does the autoencoder use backpropagation for training?**

**Answer: Yes — exactly the same as any neural network.**

**The 4-step training loop (runs once per batch):**

| Step | What happens | PyTorch call |
|---|---|---|
| 1. Forward pass | Input X flows through encoder → bottleneck → decoder → produces reconstruction X̂ | `x_hat = autoencoder(x)` |
| 2. Compute loss | MSE between original and reconstruction: L = ‖X − X̂‖² | `loss = criterion(x_hat, x)` |
| 3. Backward pass | PyTorch computes ∂L/∂w for every weight in every layer | `loss.backward()` |
| 4. Weight update | Every weight nudged to reduce loss: w = w − lr × gradient | `optimizer.step()` |

**Key difference from a classifier:**
- Classifier: compares prediction vs external label
- Autoencoder: compares reconstruction X̂ vs original input X — **the input IS the target**
- No labels needed. `TensorDataset(X_tensor, X_tensor)` — same array for both input and target

**What backpropagation does in the autoencoder specifically:**
- Loss starts at the output (reconstruction error)
- Gradients flow BACKWARDS: output → decoder → bottleneck → encoder → input
- Every single weight in all 6 layers gets a gradient computed simultaneously
- PyTorch's `autograd` engine does all of this automatically from one call: `loss.backward()`

**Why this makes it an anomaly detector:**
- After thousands of batches of BENIGN traffic, all weights adjust to reconstruct normal patterns very well → low loss
- When an ATTACK flow arrives at inference, the weights have never optimized for it → high reconstruction error
- That high error is the anomaly signal

**Complete training loop code:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap data — input and target are the SAME array
X_tensor = torch.tensor(X_train_np).to(device)
dataset  = TensorDataset(X_tensor, X_tensor)
loader   = DataLoader(dataset, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

autoencoder.train()
for batch_X, batch_target in loader:

    # Step 1: Forward pass
    reconstructed = autoencoder(batch_X)

    # Step 2: Compute loss
    loss = criterion(reconstructed, batch_target)

    # Step 3: Backpropagation
    loss.backward()

    # Step 4: Weight update + reset gradients
    optimizer.step()
    optimizer.zero_grad()
```

> **Critical reminder:** Always call `optimizer.zero_grad()` after `optimizer.step()`. If you forget, gradients accumulate across batches and produce completely wrong weight updates.

---

### 📅 16 June 2026 — Model Size Analysis

**Question answered: How large will autoencoder.pt and dqn_agent.pt be after training?**

**How model size is calculated:**
- Every connection between neurons = 1 parameter (weight or bias)
- Each parameter stored as `float32` = 4 bytes
- File size = (total parameters × 4 bytes) + small PyTorch metadata overhead

**Autoencoder parameter breakdown (115 → 64 → 32 → 16 → 32 → 64 → 115):**

| Layer | Dimensions | Weights | Biases | Total Params |
|---|---|---|---|---|
| Encoder Layer 1 | 115 → 64 | 7,360 | 64 | 7,424 |
| Encoder Layer 2 | 64 → 32 | 2,048 | 32 | 2,080 |
| Bottleneck | 32 → 16 | 512 | 16 | 528 |
| Decoder Layer 1 | 16 → 32 | 512 | 32 | 544 |
| Decoder Layer 2 | 32 → 64 | 2,048 | 64 | 2,112 |
| Output Layer | 64 → 115 | 7,360 | 115 | 7,475 |
| **TOTAL** | — | **19,840** | **323** | **20,163** |

**DQN agent parameter breakdown (115 → 128 → 64 → 5):**

| Layer | Dimensions | Total Params |
|---|---|---|
| Hidden Layer 1 | 115 → 128 | 14,848 |
| Hidden Layer 2 | 128 → 64 | 8,256 |
| Output Layer | 64 → 5 | 325 |
| **TOTAL** | — | **23,429** |

**Complete model artefact sizes:**

| File | Parameters | Estimated Size | Notes |
|---|---|---|---|
| autoencoder.pt | 20,163 | ~85–100 KB | Smaller than a JPEG image |
| dqn_agent.pt | 23,429 | ~95–115 KB | Very lightweight |
| scaler.pkl | 115 × 2 values | ~7 KB | min/max per feature |
| threshold.json | 1 value | < 1 KB | e.g. `{"threshold": 0.045}` |
| **Total artefacts** | — | **~200 KB** | Everything FastAPI loads at startup |

**Key insight:** The entire production ML system — both models, the scaler, and the threshold — fits in ~165 KB total. FastAPI loads all of this in under 1 second at startup.

**Comparison with other models:**
- Your autoencoder: ~90 KB
- ResNet-50 (image classifier): 98 MB (1,090× larger)
- BERT Base (NLP): 440 MB (4,900× larger)
- GPT-2 Small: 548 MB (6,100× larger)

> Small model size is expected and correct for tabular data. 115 input features requires far fewer parameters than images (150,528 pixels) or text (vocabulary of 50,000 tokens).

**How to verify size after training:**
```python
import torch, os

torch.save(autoencoder.state_dict(), "models/autoencoder.pt")

# Check file size
size_bytes = os.path.getsize("models/autoencoder.pt")
print(f"File size: {size_bytes / 1024:.1f} KB")

# Count parameters
total_params = sum(p.numel() for p in autoencoder.parameters())
print(f"Total parameters: {total_params:,}")

# Per-layer breakdown
for name, param in autoencoder.named_parameters():
    print(f"  {name:30s}  {str(param.shape):20s}  {param.numel():,} params")
```

---

### 📅 17 June 2026 — Split Preprocessing Pipeline (Benign + Attack)

**Topics covered:**
- Why a single combined preprocessing script is wrong for a two-stage autoencoder + DQN architecture
- How data leakage can silently happen if the MinMaxScaler is re-fitted on attack data
- Designing two purpose-built scripts with a strict run-order dependency

**Key decisions made:**

#### Problem with the original `preprocess.py`
The original script mixed benign and attack rows together, used `StandardScaler` (wrong for autoencoders), and produced a generic train/val/test split. This is incorrect for a two-model architecture where:
- The **autoencoder** must train ONLY on benign traffic to learn what "normal" looks like
- The **DQN agent** needs attack traffic to define the simulation environment states

#### `preprocess_benign.py` — Autoencoder Training Data
- Reads from `data/cicids2017/benign_data/` (5 files: monday → friday)
- Applies full CICIDS2017 cleaning pipeline:
  1. Strip column name whitespace (CICIDS2017-specific gotcha: `' Label'` ≠ `'Label'`)
  2. Drop identifier columns (`flow_id`, `timestamp`, `src_ip`, `dst_ip`)
  3. Replace `inf` / `-inf` → `NaN`, then drop NaN rows
  4. Remove exact duplicate rows (run after NaN fix, not before)
  5. Clip physics-impossible negatives to 0 (`active_*`, `packet_IAT_*`, etc.)
- **Fits** `MinMaxScaler` on training split ONLY — the scaler never sees val/test/attack data
- Splits 70% train | 15% val | 15% test (all benign — no stratification needed)
- Saves `scaler.pkl` to both `data/processed/` and `models/`
- Saves `feature_names.json` so the attack script aligns columns in the same order
- **Outputs:** `X_train_benign.npy`, `X_val_benign.npy`, `X_test_benign.npy`

#### `preprocess_attack.py` — DQN Agent Environment Data
- Reads from `data/cicids2017/attack_data/` (13 attack files)
- Applies the same cleaning steps as the benign pipeline
- Extracts and preserves string labels (`'DoS Hulk'`, `'PortScan'`, etc.) before dropping
- Encodes attack type strings → integers via `LabelEncoder`
- **Loads** (never re-fits) `scaler.pkl` produced by `preprocess_benign.py`
  - Re-fitting here would be data leakage: the scaler must only have seen benign training data
  - Attack features can legitimately fall slightly outside `[0, 1]` after this transform — that is expected and correct
- **Outputs:** `X_attacks.npy`, `y_attacks.npy`, `y_attacks_str.npy`, `attack_label_map.json`, `attack_class_counts.json`

**Why MinMaxScaler (not StandardScaler)?**
- The autoencoder uses `sigmoid` activation in the output layer → output spans `[0, 1]`
- Input features must also be in `[0, 1]` for the reconstruction loss to be meaningful
- `StandardScaler` produces z-scores (can be negative, unbounded) — wrong for this use case

**Data leakage rule confirmed:**
```
SCALER FIT  → only X_train (benign)
SCALER TRANSFORM → X_val, X_test, X_attacks (no fit, no leakage)
```

**`preprocess.py` removed** — superseded by the two split scripts above. Using a single combined script would have contaminated the autoencoder training set and used the wrong scaler type.

---

### 📅 18 June 2026 — Autoencoder Training Script Implementation

**Topics covered:**
- Implemented the complete Stage 1 training pipeline (`training/train_autoencoder.py`)
- Installed PyTorch 2.6.0+cu124 in the project virtual environment
- Pinned all project dependencies in `requirements.txt`

**`training/train_autoencoder.py` implemented:**

#### Model Architecture — `Autoencoder` (nn.Module)

| Component | Layers | Activation |
|---|---|---|
| Encoder | 115 → 64 → 32 → 16 | ReLU + Dropout(0.2) after each hidden layer |
| Bottleneck | 16 dimensions | ReLU |
| Decoder | 16 → 32 → 64 → 115 | ReLU (hidden), **Sigmoid** (output) |

- Sigmoid output is mandatory: inputs are MinMax-scaled to `[0, 1]`, so the reconstruction target must also span `[0, 1]`
- Dropout in encoder only; disabled automatically in `model.eval()` mode
- Total trainable parameters: **20,163**

#### Training Loop
- **Loss:** `nn.MSELoss()`
- **Optimizer:** `Adam(lr=1e-3, weight_decay=1e-5)`
- **Batch size:** 256 | **Max epochs:** 50
- `optimizer.zero_grad()` called **after** `optimizer.step()` (not before)
- **Early stopping:** patience = 5 epochs on `val_loss`; best weights deep-copied and restored via `copy.deepcopy(model.state_dict())`
- Train loss and val loss printed per epoch

#### Post-training threshold computation
```python
# Per-sample MSE on val set (one scalar per row, not averaged)
per_sample_mse = ((x_recon - x_batch) ** 2).mean(dim=1)

# 95th-percentile → anomaly threshold
threshold = float(np.percentile(errors, 95))
```
- A flow whose reconstruction error exceeds this threshold at inference is flagged as an anomaly

#### Outputs saved to `models/`

| File | Contents |
|---|---|
| `autoencoder.pt` | `state_dict` only (not the full model object) |
| `threshold.json` | `{"threshold": <float>}` — 95th-percentile val MSE |
| `training_history.json` | `{"train_loss": [...], "val_loss": [...]}` — one float per epoch |

**Code style decisions:**
- Path anchoring via `os.path.dirname(os.path.abspath(__file__))` → project root (matches `preprocess_benign.py`)
- UTF-8-safe console + file logging (same handler setup as preprocessing scripts)
- Step-numbered `[STEP N]` log lines throughout
- `main()` entry point; all logic in named, docstring-annotated functions

**Environment setup completed:**
- Installed `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `torchaudio==2.6.0+cu124`
- GPU confirmed: NVIDIA GeForce GTX 1650, driver 592.27, CUDA 12.x
- All 24 packages pinned in `requirements.txt`

**What remains for Stage 1:**
- Run `python training/train_autoencoder.py` once preprocessing is complete
- Verify `threshold.json` and `autoencoder.pt` are produced correctly
- Plot `training_history.json` loss curves to confirm convergence

---

### 📅 18 June 2026 — Frontend Dashboard Initiation (ThreatSentinel UI)

**Topics covered:**
- Scaffolded the React/TypeScript incident dashboard using Vite
- Integrated Tailwind CSS v4 with the `@tailwindcss/vite` plugin
- Built the initiation UI (`ThreatSentinel`) — a dark cybersecurity dashboard scaffold

**Tech stack set up:**

| Tool | Version | Role |
|---|---|---|
| React | 19.2.6 | UI framework |
| TypeScript | 6.0.2 | Type safety |
| Vite | 8.0.16 | Dev server + bundler |
| Tailwind CSS | 4.3.1 | Utility-first styling |
| `@tailwindcss/vite` | 4.3.1 | Tailwind v4 Vite plugin |

**Project location:** `frontend/incident-dashboard/`

**Tailwind v4 wiring (Vite plugin approach — no `tailwind.config.js` needed):**
```typescript
// vite.config.ts
import tailwindcss from '@tailwindcss/vite'
export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```
```css
/* index.css */
@import "tailwindcss";
```

**Design system implemented (`index.css`):**
- Dark cybersecurity colour palette: `--col-bg: #080b14`, `--col-cyan: #38bdf8`, `--col-red: #f87171`, `--col-green: #34d399`
- Typography: Inter (body) + JetBrains Mono (terminal/code text) via Google Fonts
- Animated CSS: grid background, ambient glow orbs, pulse ring, scan line, blink cursor

**Dashboard components built (`App.tsx`):**

| Component | Description |
|---|---|
| **Navbar** | Logo, model status pill (`MODEL NOT TRAINED`), nav links |
| **Stat cards** | Flows Analysed, Threats Detected, Autoencoder Loss, Threshold |
| **Pipeline tracker** | 5-stage progress list (Data Ingestion → Live Inference) with status dots |
| **Live threat feed** | Tabular alert list + WebSocket connection status + “awaiting model” banner |
| **Architecture breadcrumb** | Full inference pipeline visualised inline as styled tokens |
| **Footer** | Stack versions, project attribution |

**Boilerplate removed:**
- Vite counter button, React/Vite logos, hero image
- `App.css` scaffold styles (`.hero`, `.ticks`, `#next-steps`, `.counter`)
- `index.css` Vite light/dark default theme

**Status:** Dashboard serves as a placeholder scaffold. All data is mock/static.
Live data will be wired via `ws://localhost:8000/ws` WebSocket once the FastAPI backend is implemented.

**Dev server running at:** `http://localhost:5174/`

---

### 📅 18 June 2026 — Replay Simulator Design (Demo Infrastructure)

**Topics covered:**
- Designed the full architecture of the network log replay simulator
- Defined the responsibility split between simulator, backend, and dashboard
- Identified the `simulator/` module as a new top-level project component

**The problem being solved:**
The project does not have access to a live network tap or real router/firewall hardware (explicitly out of scope in the project charter: *"Integration with real firewall or router hardware"* and *"Real-time packet capture (PCAP)"* are excluded). The replay simulator bridges this gap for demonstration and validation purposes.

**What the replay simulator does:**

```
Replay Simulator
    │
    │  1. Load a pool of rows from raw CICIDS2017 CSVs at startup
    │     (mix of benign + attack, keeping src_ip/dst_ip/true_label locally)
    │
    │  2. Every N seconds (configurable), pick one row at random
    │     and POST its 115 feature values (unscaled) to /predict
    │
    │  3. Print ground-truth vs prediction comparison locally
    │     (running precision/recall sanity check during demo)
    ▼
FastAPI /predict
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Read from raw CSVs, not `.npy` | `.npy` files strip `src_ip`, `dst_ip`, true label — needed for display and sanity checking |
| Send features **unscaled** | Scaling is backend's responsibility (matches architecture doc); simulator behaves like a real flow collector |
| True label stays **client-side** | Never sent to backend — preserves fair evaluation; used for live precision/recall printing |
| Sample a fixed pool at startup | Avoids re-reading 1.8 GB of CSVs per request; same cost as preprocessing |
| `time.sleep(random 0.2–2s)` | Makes flow arrival feel live; configurable via `--rate` flag |
| `--benign-ratio` flag | Controls attack frequency — lower for dramatic demo, higher for realistic baseline |

**Dummy action executor (backend side):**
- No real firewall or router calls
- Maintains an in-memory `simulated_network_state` dict: `{ "192.168.1.45": { "status": "blocked", "action": "Block IP" } }`
- When DQN selects an action, the backend updates this dict and broadcasts via WebSocket
- Dashboard displays per-IP status badges live: 🟢 active / 🔴 blocked / 🟡 isolated / 🔵 monitored

**5 remediation actions the DQN can select:**

| Action | Simulated effect |
|---|---|
| Block IP | Source IP status → `blocked` |
| Revoke credentials | Session status → `revoked` |
| Isolate server | Destination status → `isolated` |
| Kill process | Process status → `terminated` |
| Monitor (no action) | No state change; incident logged only |

**New `simulator/` module added to project structure:**
- `simulator/replay_simulator.py` — main script (to be written next)
- `simulator/README.md` — usage instructions (to be written next)

**Status:** Folder scaffold created. Code to be written in the next session.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING                         │
│                                                             │
│  CICIDS2017 CSVs ──► Preprocessing ──► Autoencoder         │
│  UNSW-NB15 CSVs      (Pandas/NumPy)    (PyTorch+CUDA)      │
│                           │                  │              │
│                      scaler.pkl       autoencoder.pt        │
│                      threshold.json   dqn_agent.pt          │
└──────────────────────────┬──────────────────┘──────────────┘
                           │ load at startup
┌──────────────────────────▼──────────────────────────────────┐
│                    ONLINE PRODUCTION                        │
│                                                             │
│  Network Flow ──► FastAPI ──► Autoencoder ──► DQN Agent    │
│                      │              │              │        │
│                 WebSocket     threshold?      Block/Isolate │
│                      │                            │         │
│               React Dashboard ◄──── PostgreSQL ◄──┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data | Pandas 2.x | CSV loading, cleaning, filtering |
| Data | NumPy 1.26+ | Array math, normalization |
| Data | Scikit-learn 1.4+ | MinMaxScaler, train/test split |
| Data | PostgreSQL 15+ | Persistent incident storage |
| ML | PyTorch 2.x + CUDA 12 | Define and train both models |
| ML | Gymnasium 0.29+ | DQN simulation environment |
| ML | Matplotlib 3.x | Training diagnostics |
| Backend | FastAPI 0.110+ | REST API + WebSocket server |
| Backend | SQLAlchemy 2.x | ORM for PostgreSQL |
| Backend | Uvicorn 0.29+ | ASGI server |
| Frontend | React.js 18+ | Dashboard UI |
| Frontend | Recharts 2.x | Live charts |
| DevOps | Docker 24+ | Container packaging |
| DevOps | Git | Version control |

---

## Datasets

### CICIDS2017
- **Source:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Kaggle:** https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Size:** ~2.8 million flow records, 78 features, 14 attack types
- **Usage:** Primary training and evaluation dataset

### UNSW-NB15
- **Source:** Australian Centre for Cyber Security (ACCS), UNSW Canberra
- **Kaggle:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- **Size:** ~2.5 million records, 49 features, 9 attack types
- **Usage:** Cross-dataset generalization evaluation

---

## Project Milestones

- [x] AI Opportunity Identification & Feasibility *(11–13 Jun 2026)*
- [x] AI Project Planning & Architecture Design *(13–15 Jun 2026)*
- [ ] Data Readiness & Engineering *(15–21 Jun 2026)*
- [ ] AI Model Prototyping *(21–29 Jun 2026)*
- [ ] Model Development & Optimization *(29 Jun–08 Jul 2026)*
- [ ] AI Solution Integration & Deployment *(08–17 Jul 2026)*
- [ ] AI Validation, Governance & Release Readiness *(17–22 Jul 2026)*
- [ ] Production Monitoring & Continuous Improvement *(22–24 Jul 2026)*
- [ ] Project Closure & Handover *(24–25 Jul 2026)*

---

## Research References

### Dataset Papers
1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP 2018.
   - 🔗 https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

2. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*. MilCIS 2015.
   - 🔗 https://ieeexplore.ieee.org/document/7348942

### Autoencoder & Anomaly Detection
3. *Analysis of Autoencoders for Network Intrusion Detection* (2021). PMC/NCBI.
   - 🔗 https://pmc.ncbi.nlm.nih.gov/articles/PMC8272075/

4. Abdalla, A. et al. (2023). *Enhancing Network Intrusion Detection Using Attention-Based Deep Autoencoders*. Expert Systems with Applications, 213, 119102.
   - 🔗 https://www.sciencedirect.com/science/article/pii/S0957417422019911

5. *Unsupervised Machine Learning Methods for Anomaly Detection in Network Packets* (2025). Electronics, MDPI.
   - 🔗 https://www.mdpi.com/2079-9292/14/14/2779

6. *Feature Importance Guided Autoencoder for Dimensionality Reduction in Intrusion Detection Systems* (2026). Scientific Reports.
   - 🔗 https://www.nature.com/articles/s41598-026-36695-9

7. *Deep learning-driven methods for network-based intrusion detection systems: A systematic review* (2025). ScienceDirect.
   - 🔗 https://www.sciencedirect.com/science/article/pii/S2405959525000050

8. *A deep learning/machine learning approach for anomaly based network intrusion detection* (2025). Frontiers in AI.
   - 🔗 https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1625891/full

### Deep Q-Network (DQN)
9. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602.
   - 🔗 https://arxiv.org/abs/1312.5602

10. *Deep Q-Learning based Reinforcement Learning Approach for Network Intrusion Detection*. arXiv:2111.13978.
    - 🔗 https://arxiv.org/pdf/2111.13978

### Reinforcement Learning for Cyber Defense
11. Nguyen, T. T., & Reddi, V. J. (2023). *Deep Reinforcement Learning for Cyber Security*. IEEE Transactions on Neural Networks and Learning Systems, 34(8), 3779–3795.
    - 🔗 https://research.monash.edu/en/publications/deep-reinforcement-learning-for-cyber-security/

12. *Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey* (2024). arXiv:2310.07745.
    - 🔗 https://arxiv.org/html/2310.07745v2

13. Ma, Y. et al. (2024). *Application of deep reinforcement learning algorithms for automatic threat detection and response in dynamic network environments*. SAGE Journals.
    - 🔗 https://journals.sagepub.com/doi/abs/10.1177/14727978241309550

14. Sewak, M., Sahay, S. K., & Rathore, H. (2023). *Deep Reinforcement Learning in the Advanced Cybersecurity Threat Detection and Protection*. Information Systems Frontiers, 25(2), 589–611.
    - 🔗 https://link.springer.com/article/10.1007/s10796-022-10333-x

---

## Project Structure

```
project/
├── data/
│   ├── cicids2017/
│   │   ├── benign_data/         # 5 benign CSVs (monday → friday)
│   │   └── attack_data/         # 13 attack CSVs (dos_hulk, portscan, etc.)
│   ├── unsw_nb15/               # raw UNSW-NB15 CSV files
│   └── processed/               # output of preprocessing scripts
│       ├── X_train_benign.npy   # autoencoder training input
│       ├── X_val_benign.npy     # autoencoder validation input
│       ├── X_test_benign.npy    # autoencoder test input
│       ├── X_attacks.npy        # DQN environment states (attack flows)
│       ├── y_attacks.npy        # integer-encoded attack type labels
│       ├── y_attacks_str.npy    # string attack type labels
│       ├── scaler.pkl           # fitted MinMaxScaler (also in models/)
│       ├── feature_names.json   # ordered feature column list
│       └── attack_label_map.json
├── simulator/               # CICIDS2017 replay simulator (demo infrastructure)
│   ├── replay_simulator.py  # streams CSV rows to /predict as if live traffic
│   └── README.md            # usage: --rate, --benign-ratio, --host flags
├── notebooks/               # Jupyter notebooks for EDA
├── training/
│   ├── preprocess_benign.py # Step 1 — benign data → autoencoder training data
│   ├── preprocess_attack.py # Step 2 — attack data → DQN environment data
│   ├── train_autoencoder.py # Stage 1 training
│   └── train_dqn.py         # Stage 2 training
├── models/                  # saved artefacts (output of training)
│   ├── autoencoder.pt          # ~90 KB  (state_dict, 115-feature model)
│   ├── dqn_agent.pt            # ~95 KB
│   ├── scaler.pkl              # ~7 KB   (copy from data/processed/)
│   ├── threshold.json          # <1 KB   (95th-percentile val MSE)
│   └── training_history.json  # train/val loss per epoch
├── backend/
│   ├── main.py              # FastAPI app + WebSocket
│   ├── inference.py         # model inference logic
│   ├── action_executor.py   # dummy action executor (simulated network state)
│   ├── database.py          # SQLAlchemy + PostgreSQL
│   └── schemas.py           # Pydantic models
├── frontend/
│   └── incident-dashboard/  # Vite+React 19+TypeScript+Tailwind v4
│       ├── src/
│       │   ├── App.tsx          # ThreatSentinel dashboard UI
│       │   ├── App.css          # component styles
│       │   ├── index.css        # design system + Tailwind
│       │   └── main.tsx         # React entry point
│       ├── index.html
│       ├── vite.config.ts
│       └── package.json
├── docs/
│   ├── architecture_document.docx
│   └── project_charter.docx
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/incident-tracking-remediation.git
cd incident-tracking-remediation

# 2. Install Python dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# 3. Preprocess benign data (fits MinMaxScaler, produces autoencoder training arrays)
#    Output: X_train/val/test_benign.npy, scaler.pkl, feature_names.json
python training/preprocess_benign.py

# 4. Preprocess attack data (loads scaler, produces DQN environment arrays)
#    Output: X_attacks.npy, y_attacks.npy, attack_label_map.json
#    NOTE: Must run AFTER preprocess_benign.py (depends on scaler.pkl)
python training/preprocess_attack.py

# 5. Train the autoencoder (produces autoencoder.pt + threshold.json)
python training/train_autoencoder.py

# 6. Train the DQN agent (produces dqn_agent.pt)
python training/train_dqn.py

# 7. Start the backend
uvicorn backend.main:app --reload

# 8. Start the frontend (from frontend/incident-dashboard/)
cd frontend/incident-dashboard && npm install && npm run dev

# 9. Run the replay simulator (streams CICIDS2017 flows to backend as live traffic)
#    --rate: flows per second | --benign-ratio: fraction of benign rows (0.0–1.0)
#    NOTE: backend must be running before starting the simulator
python simulator/replay_simulator.py --rate 1 --benign-ratio 0.8
```

---

*README last updated: 18 June 2026 (evening — replay simulator design + folder scaffold)*
*Next update due: When backend /predict and simulator code are written*