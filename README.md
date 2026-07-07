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

### 📅 19 June 2026 — End-to-End Autoencoder Evaluation & Hybrid Architecture Decision

**Context:** Evaluated the original Autoencoder (v1) and identified a hard ceiling for specific attack types, motivating an ablation study on bottleneck capacity and a pivot towards a multi-stage hybrid classifier approach. *(Note: This work falls under early Model Prototyping work pulled forward).*

**Topics covered:**
- Trained the autoencoder (v1) with the original 115→64→32→16→32→64→115 architecture; got F1=0.6967 on initial local test, TPR=0.56 (well below the >0.90 recall target from the project charter).
- Diagnosed the gap via threshold sweeping (`eval_threshold.py`) using `precision_recall_curve` on the full benign val set + full attack set. Found the naive 95th-percentile threshold was too conservative, but also found that a "best F1" threshold from the skewed dataset ratio was operationally useless (FPR=0.88 — would flag most benign traffic).
- Built `tests/test_autoencoder.py` with a realistic percentile-based threshold sweep (based on benign-only error percentiles); selected p92.5 (threshold=0.00071623) as the practical operating point — Recall=0.7145, Precision=0.9050, F1=0.7985, FPR=0.075.
- Built `training/analyze_feature_errors.py` to diagnose WHY 6 specific attack classes (SSH-Patator, Web_Brute_Force, Web_XSS, Web_SQL_Injection, FTP-Patator, Botnet_ARES) were poorly detected. Found they split into two groups:
  - **FTP-Patator and Botnet_ARES** had real but compressed separating signal (12 and 14 significant features respectively, comparable to DoS_Hulk's 19).
  - **SSH-Patator, Web_Brute_Force, Web_XSS, Web_SQL_Injection** had 0-1 significant features — i.e. genuinely indistinguishable from benign at the flow-feature level.
- Ran an ablation: trained autoencoder v2 with a wider bottleneck (115→80→48→24→48→80→115, up from 16-dim) to test whether more bottleneck capacity would recover FTP-Patator/Botnet_ARES detection.
- Compared v1 vs v2 (`training/compare_models.py`): v2 won on aggregate (F1 0.7136→0.8217, Recall 0.58→0.71, same FPR=0.05) and dramatically improved Port_Scan (0.51→0.99 TPR) as an unexpected bonus, but did NOT recover FTP-Patator (0.2658→0.2656, flat) or Botnet_ARES (0.0372→0.0369, flat) — confirming these two classes hit a feature-representation ceiling, not a model-capacity ceiling.
- Promoted v2 to be the new primary model: archived v1 files to `models/archive/` (`autoencoder_v1_16dim.pt`, `threshold_v1_16dim.json`, `training_history_v1_16dim.json`), updated `models/autoencoder.pt`, `threshold.json`, `training_history.json` to the v2 (24-dim) versions, and updated `training/train_autoencoder.py` to use the 24-dim architecture as the new canonical definition going forward.

**Key decisions made:**
- Confirmed the reconstruction-error architecture has a real ceiling for 5 attack types (FTP-Patator, Botnet_ARES, SSH-Patator, Web_Brute_Force, Web_XSS) — these attacks statistically resemble benign traffic at the 115-feature flow level, and no amount of threshold tuning or bottleneck widening recovers them.
- Web_SQL_Injection (n=24) and Heartbleed (n=12) remain flagged as statistically unevaluable due to sample size, consistent with the existing Heartbleed caveat already documented in this README.
- Decided NOT to pursue 5 separate per-class binary classifiers; will instead build ONE multi-class hybrid classifier (XGBoost) covering just the 5 confirmed weak classes, to keep the production pipeline at low latency (<50ms target) and avoid 5x inference calls per flow.
- v2 (24-dim) autoencoder is now the primary model in production; v1 (16-dim) retained in `models/archive/` for the evaluation report's ablation section.

**Next Steps:**
- Build `training/build_hybrid_dataset.py` — construct a labeled dataset of the 5 weak attack classes + a matched benign sample, stratified train/val/test split.
- Build `training/train_hybrid_classifier.py` — train a class-weighted multi-class XGBoost classifier on this dataset.
- Build `training/evaluate_combined_pipeline.py` — evaluate the FULL two-stage pipeline (autoencoder v2 + hybrid classifier together) end to end on the full attack/benign test data, comparing combined metrics against the autoencoder-alone baseline.
- This hybrid classifier becomes Stage 1B: flows that pass the autoencoder threshold as "normal" get a second check from the hybrid classifier before being confirmed benign.
- Target: push overall Recall closer to the charter's >0.90 goal by catching these 5 classes via supervised signal instead of reconstruction error.

---

### 📅 22 June 2026 — Deep Dive into Preprocessing and Autoencoder Model

**Topics covered:**
- Reviewed and understood the data preprocessing pipelines: `preprocess_benign.py` and `preprocess_attack.py`.
- Studied the Autoencoder model implementation (`train_autoencoder.py`), its architecture, and potential improvements.
- Watched the [Deep Learning Tutorial video](https://www.youtube.com/watch?v=CAgWNxlmYsc&list=PLKnIA16_Rmvboy8bmDCjwNHgTaYH2puK7&index=5) to solidify understanding of the model's concepts and mechanics.
- Used ChatGPT as an interactive learning assistant to clarify concepts. Logged the session here: [ChatGPT Conversation](https://chatgpt.com/share/6a393fe9-132c-83ee-8a48-04c924a693b6).

**Key takeaways:**
- Gained a solid grasp of how benign traffic is used to train the autoencoder, and how attack traffic is kept isolated for DQN environment states.
- Understood the training loop, loss function computation, and threshold generation process.
- Explored ways to further improve the autoencoder model based on insights from the tutorial and chat discussions.

---

### 📅 23 June 2026 — DoS_Hulk False-Negative Diagnosis & Hybrid Classifier Extension to 7 Classes

**Context:** After completing `evaluate_combined_pipeline.py`, DoS_Hulk was identified as the dominant source of false negatives in the combined two-stage pipeline: 297,642 total samples, TPR = 0.5194, 143,056 flows missed — accounting for ~95% of all remaining false negatives.

---

#### Part 1 — Root-Cause Diagnosis (`training/diagnose_dos_hulk.py`)

Built a 4-step diagnostic script to answer: **should we lower the AE threshold, or extend the hybrid classifier?**

**STEP 1 — Error distribution analysis:**
- Loaded `errors_attack.npy` + `y_attacks_str.npy`, filtered to DoS_Hulk
- Key finding: threshold sits at the **48th percentile** of DoS_Hulk errors — nearly half the class has very low reconstruction error
- Missed-flow bucket split:

| Bucket | Count | % of missed |
|---|---|---|
| Near threshold `[0.5×T, T]` | 4,414 | 3.09% |
| Far below `[0, 0.5×T)` | 138,642 | 96.91% |

- **Verdict: LIKELY REQUIRES FEATURE/MODEL FIX, NOT THRESHOLD** — 96.9% of missed DoS_Hulk flows sit far below even half the threshold, meaning they are nearly indistinguishable from benign traffic as far as the autoencoder is concerned. Lowering the threshold enough to catch them would flood the pipeline with false positives.

**STEP 2 — Threshold sensitivity sweep (always runs):**

| Threshold | Hulk TPR | Overall TPR | Benign FPR | Note |
|---|---|---|---|---|
| 0.0009142810 | 0.5194 | 0.7121 | 0.0500 | CURRENT |
| 0.0007 | 0.5286 | 0.7171 | 0.0633 | FPR≤0.10 |
| 0.0005 | 0.5340 | 0.7211 | 0.0944 | FPR≤0.10 |
| 0.0003 | 0.5504 | 0.7415 | 0.1483 | FPR>0.10 ⚠️ |
| 0.0002 | 0.5802 | 0.7597 | 0.2077 | FPR>0.10 ⚠️ |
| 0.0001 | 0.5802 | 0.7690 | 0.3020 | FPR>0.10 ⚠️ |

- Confirmed quantitatively: the maximum Hulk TPR gain within the FPR ≤ 0.10 budget is only **+3.4 pp** (0.5194 → 0.5340). The 48% of missed flows that sit far below threshold cannot be recovered by any operationally realistic threshold change.

**STEP 3 — Sub-population feature analysis (ran because verdict = FEATURE/MODEL FIX):**
- Split DoS_Hulk into detected (154,586) vs missed (143,056) at the current threshold
- Computed mean feature values in each sub-population across all 115 features
- Top 15 features by absolute mean difference (`|Δ|`) revealed clear structural separation
- **Separation ratio = avg top-15 |Δ| / avg feature mean ≥ 0.20** → classified as **CLEAR FEATURE SEPARATION**
- Interpretation: missed DoS_Hulk flows exhibit systematically different feature values from detected flows — they represent a **distinguishable sub-variant** of DoS_Hulk that the autoencoder reconstructs accurately (hence low error), making them invisible to Stage 1.

**STEP 4 — Recommendation issued:**
> **HYBRID CLASSIFIER EXTENSION RECOMMENDED:** missed DoS_Hulk forms a distinguishable sub-population based on Step 3 features; add `DoS_Hulk` as a 7th class to the existing hybrid classifier, training only on the currently-missed rows as positive examples.

---

#### Part 2 — Hybrid Classifier Extension: Adding DoS_Hulk as 7th Class

**Conceptual design:**
- The hybrid classifier (Stage 1B) already handles 5 weak attack classes + Benign (6 classes total)
- Extend it to handle DoS_Hulk as a **7th class**, but using ONLY the 143,056 missed flows as positive training examples
- Detected DoS_Hulk flows (already caught by AE) must NOT be included — they would be double-counted
- Class imbalance (143K DoS_Hulk vs ~950 Web_XSS) handled downstream by `compute_sample_weight('balanced')` in `train_hybrid_classifier.py` — no code change needed there

**Key implementation insight — AE-error filtering:**
The filter `y_all == 'DoS_Hulk' AND errors_all ≤ threshold` is the critical correctness constraint. It ensures only the flows the autoencoder **currently passes** as normal are trained as positive DoS_Hulk examples. Using all DoS_Hulk rows would mix in the already-detected flows and create training data that doesn't match the inference-time distribution.

**`build_hybrid_dataset.py` changes (only file modified):**

| Change | Rationale |
|---|---|
| `MODEL_DIR` path constant added | Needed to load `models/threshold.json` |
| `INCLUDE_DOS_HULK_MISSED = True` toggle | One flag to revert to 6-class model if needed |
| `DOS_HULK_CLASS = "DoS_Hulk"` | Class name for label assignment |
| `DOS_HULK_MAX_SAMPLES = None` | Use all 143K rows; set int to sub-sample |
| `step1_load_attack_subset()` now returns `X_all, y_all, errors_all` too | Avoids reloading the 600K array in a separate call |
| New `step1b_load_dos_hulk_missed()` | Reads threshold.json, masks DoS_Hulk AND error ≤ threshold, returns only missed rows |
| `step3_combine_and_split()` accepts optional `X_hulk_missed, y_hulk_missed` | Appends before concat/split; `None` = 6-class mode unchanged |
| `del X_all, errors_all` after step 1B | Frees ~530 MB RAM before loading X_train_benign (1.1 GB) |

**`train_hybrid_classifier.py` — zero changes:** dynamically reads `n_classes` from `hybrid_label_map.json`, so it automatically handles 7 classes.

**`evaluate_combined_pipeline.py` — zero changes:** already handles any number of hybrid classes.

**Dataset composition after extension:**

| Class | Rows (approx.) | Source |
|---|---|---|
| Benign | 25,000 | X_train_benign.npy (sampled) |
| FTP-Patator | 9,531 | X_attacks.npy (5-class filter) |
| Botnet_ARES | 5,508 | X_attacks.npy |
| SSH-Patator | 5,949 | X_attacks.npy |
| Web_Brute_Force | 2,733 | X_attacks.npy |
| Web_XSS | 1,357 | X_attacks.npy |
| **DoS_Hulk (missed)** | **143,056** | **X_attacks.npy filtered by errors_attack.npy ≤ threshold** |
| **Total** | **~193,134** | |

**Pipeline fix resolved today:** XGBoost 3.x API breaking change — `early_stopping_rounds` was moved from `fit()` to the `XGBClassifier` constructor. Fixed by moving the parameter into `XGB_PARAMS` dict.

**Diagnostics fix resolved today:** Float key precision mismatch in `diagnose_dos_hulk.py` — `THRESHOLD_CANDIDATES` hardcoded `0.0009142810` (literal) but `json.load()` returns `0.0009142810013145208` (full precision), causing `KeyError` on dict lookup. Fixed by building the sweep list at runtime, prepending `current_threshold` (the exact JSON-loaded value) as the first element.

**Scripts executed today (in order):**
1. `python training/build_hybrid_dataset.py` — rebuilt 7-class dataset (~193K rows)
2. `python training/train_hybrid_classifier.py` — retrained XGBoost with 7 classes
3. `python training/evaluate_combined_pipeline.py` — full two-stage pipeline evaluation

**Expected outcome of today's extension:** DoS_Hulk TPR should improve significantly from 0.52 (autoencoder alone) since the hybrid classifier now has 100K+ labeled positive examples of the missed sub-variant to train on.

**New files created today:**
- `training/diagnose_dos_hulk.py` — 4-step root-cause analysis for DoS_Hulk false negatives

**Files modified today:**
- `training/build_hybrid_dataset.py` — added Step 1B (DoS_Hulk missed flow extraction), extended to 7-class output
- `training/train_hybrid_classifier.py` — fixed XGBoost 3.x `early_stopping_rounds` API change
- `training/diagnose_dos_hulk.py` — fixed float key precision bug in threshold sweep dict lookup

---

### 📅 23 June 2026 — Architecture Extension — Attack-Type Classification + DQN Remediation Pipeline Design

**Context:** Full two-stage combined pipeline (Autoencoder v2 + Hybrid XGBoost Classifier) evaluated end-to-end on the CICIDS2017 held-out test set. Results confirmed progress and identified the next frontier.

#### Combined Pipeline Evaluation Results

| Stage | Overall TPR | Note |
|---|---|---|
| Autoencoder v2 alone | 0.7121 | 24-dim bottleneck at p92.5 threshold |
| + Hybrid XGBoost Classifier (Stage 1B) | **0.7497** | +3.76 pp lift from catching 5 weak attack classes |

The hybrid classifier (Stage 1B) successfully recovers the 5 attack types (Botnet_ARES, FTP-Patator, SSH-Patator, Web_Brute_Force, Web_XSS) that produce benign-like flow statistics and evade the autoencoder. Combined pipeline TPR is now 0.7497 vs 0.7121 for the autoencoder alone — a confirmed +3.76 pp gain.

#### DoS_Hulk Diagnostic Finding

DoS_Hulk remains the dominant source of false negatives (297,642 total samples, TPR ≈ 0.52 autoencoder-alone). Deep diagnostic analysis via `training/diagnose_dos_hulk.py` revealed:

- **48% of DoS_Hulk flows form a distinguishable low-signal sub-variant**: characterised by short/single-packet flows and near-zero backward traffic features. These flows produce reconstruction errors far below even half the anomaly threshold — they are structurally distinct from the high-signal DoS_Hulk flows the autoencoder does catch.
- **Threshold lowering cannot recover them** within any operationally realistic FPR budget: the maximum TPR gain within FPR ≤ 0.10 is only +3.4 pp (0.5194 → 0.5340), while 96.9% of missed flows sit below 0.5× the threshold.
- **Recommendation issued by diagnostic:** add DoS_Hulk as a 7th class to the hybrid classifier, training only on the currently-missed sub-population (143,056 rows) as positive examples. This was acted on today (see previous log entry for full implementation details).

> **Open item:** Even with the 7-class hybrid classifier extension, the DoS_Hulk low-signal sub-variant warrants further investigation. Future work may include targeted feature engineering (e.g., packet-count ratios, inter-arrival time statistics) or a dedicated binary classifier for this sub-population.

#### Architecture Extension Decision: 3-Stage Pipeline

After reviewing the combined pipeline results and the DoS_Hulk diagnostic, the decision was made to extend the detection system from a 2-stage to a **4-stage architecture**, built incrementally:

| Stage | Component | Purpose | Status |
|---|---|---|---|
| 1A | Autoencoder (v2, 24-dim) | Unsupervised anomaly detection on flow reconstruction error | ✅ Done |
| 1B | Hybrid XGBoost Classifier | Catches 5 weak attack classes that evade the AE | ✅ Done |
| 2 | **Attack-Type Neural Network** | Broad multi-class classifier across all 13 CICIDS2017 attack types — answers "which attack type is this flow?" | 🔨 Building today |
| 3 | **DQN Remediation Agent** | Given flow features + attack-type prediction, selects optimal remediation action from 5 choices | 🔨 Building today |
| 4 | LLM Agent Layer *(planned)* | For low-confidence / unknown attacks: generate human-readable summary, route to human review | 📋 Future work |

**Why an Attack-Type NN in addition to the hybrid classifier?**
The hybrid classifier (Stage 1B) covers only the 5 attack types with AE-evasion characteristics. The new Attack-Type NN is deliberately **broader** — it classifies across ALL attack types present in CICIDS2017 — so that the DQN agent has a rich, calibrated probability vector over attack types as part of its state, rather than only a 5-class or binary signal. This gives the DQN much better information for choosing the correct remediation action (e.g., distinguishing Block IP for DoS vs. Revoke Credentials for FTP-Patator).

**Key design constraint for the Attack-Type NN:** Heartbleed (n=12) and Web_SQL_Injection (n=24) are excluded from training due to insufficient samples. These two classes are documented as always "unclassified" by this NN — they remain covered by the autoencoder's reconstruction error signal.

#### LLM Agent Layer — Planned Future Work (Explicitly Out of Scope Today)

A Stage 4 LLM-based human-escalation layer is planned for a future session, subject to the following constraints and scope limitations:

- **Scope:** advisory and human-facing ONLY — the LLM layer will generate human-readable incident summaries and routing recommendations, and will NOT autonomously trigger any remediation actions.
- **Trigger condition:** low-confidence predictions from the Attack-Type NN (max softmax probability below a tunable threshold) OR attack types that fall outside the NN's training distribution.
- **Build sequence:** Stage 4 will be designed and built AFTER Stage 3 (DQN) is complete and results have been reviewed with the project mentor.
- **Mentor review required** before implementation: the LLM layer scope, API choice, and cost model will be discussed explicitly before any code is written.

---

### 📅 24 June 2026 — DQN Confusion Diagnosis — Hulk/FTP Robustness Confirmed, Web Attack Action Merge

**Topics covered:**

- Diagnosed two confusion patterns identified in the Attack-Type NN's held-out test confusion matrix:
  - **(a) DoS_Hulk → FTP-Patator:** 3,363 of 44,647 true DoS_Hulk test rows were misclassified as FTP-Patator (7.5% error rate on that class)
  - **(b) Web_Brute_Force ↔ Web_XSS:** 187 of 410 Web_Brute_Force rows predicted as Web_XSS (45.6%), and 104 of 204 Web_XSS rows predicted as Web_Brute_Force (51.0%) — near-random mutual confusion

- Built `evaluation/diagnose_nn_confusions.py` — initial diagnostic covering both confusion pairs:
  - For each pair: computes normalized separation scores (`|mean_A − mean_B| / pooled_std`) across all 115 features and the 8 previously-identified "quiet sub-variant" features, to distinguish fixable model/training issues from genuine feature-representation ceilings
  - Runs a pooled-std threshold filter to exclude constant-zero feature columns (35 excluded, e.g., `urg_flag_counts`, `active_*`, `idle_*`) that were producing NaN values and corrupting the ranking

- Built `evaluation/diagnose_hulk_ftp_confusion.py` — full 115-feature deep-dive on the DoS_Hulk/FTP-Patator pair, with two parts:
  - **Part A — feature separation:** for each of the 80 valid (non-constant) features, computes normalized separation between the 3,363 misclassified Hulk rows, the 1,185 correctly-classified FTP-Patator rows, and the 41,100 correctly-classified Hulk rows as reference. Result: 65 features with norm_sep > 1.0, max = 2.64 — strong separating signal exists. The misclassified Hulk rows resemble true FTP-Patator on 17 of the top 20 features, consistent with the "quiet sub-variant" (short-connection, near-zero backward traffic) profile pulling these flows across the class boundary.
  - **Part B — DQN end-to-end impact test:** constructed the exact DQN state vectors the agent would receive at real inference time for these 3,363 rows — using the NN's *actual wrong* FTP-Patator softmax output (not ground-truth) — and ran the trained DQN on them. Result: **3,362 of 3,363 rows (99.97%) still received Block IP (action 0)**, which is the correct remediation for DoS_Hulk. Only 1 row received the wrong action. The DQN's state includes the raw 115 flow features and AE reconstruction error alongside the NN's softmax, and the DQN learned during training to weight the raw features more heavily for this case, effectively overriding the NN's wrong classification.

- **Finding 1 (DoS_Hulk → FTP-Patator):** Strong separating feature signal exists — this IS a fixable model/training issue in isolation (likely class-weight overcorrection toward FTP-Patator's minority class). However, the end-to-end DQN compensation test showed zero measured operational impact: the DQN already correctly handles these rows at 99.97% accuracy despite the wrong upstream classification. CONCLUSION: do not fix this in the NN. The result is documented as a working example of the layered-state DQN design providing natural robustness against upstream classifier errors.

- **Finding 2 (Web_Brute_Force ↔ Web_XSS):** No separating signal at the feature level — max normalized separation across all 115 features = 0.198, far below even the "moderate" threshold of 0.5. This is a genuine feature-representation ceiling: these two attack types produce statistically indistinguishable flow statistics. Unlike Finding 1, the two classes had *different* ground-truth-optimal DQN actions (Revoke Credentials vs Kill Process), so the DQN had no raw-feature fallback to resolve the ambiguity. Intervention was necessary.

- **Fix applied for Finding 2:** Merged both Web_Brute_Force and Web_XSS to a single unified target action — **Revoke Credentials (action 1)** — in the ground-truth-optimal-action table (`ACTION_MAP` in `training/build_dqn_environment.py`). Reasoning: Web_Brute_Force is fundamentally a credential-based attack; Revoke Credentials is a more conservative and safer default than Kill Process when the classification is genuinely ambiguous between the two types. DQN environment arrays were regenerated and the DQN agent was retrained from scratch with the corrected reward table. No Attack-Type NN retraining was required — the merge was applied purely downstream of the NN's existing softmax output.

- Pre-merge DQN backed up to `models/archive/dqn_agent_pre_web_merge.pt` before overwriting `models/dqn_agent.pt` with the retrained version.

**Key decisions made:**

- **Adopted a general diagnostic principle for this project going forward:** when an upstream stage shows a confusion pattern, do not assume it needs fixing. Test whether the *full pipeline* (all downstream stages included) actually produces a wrong final outcome before spending engineering time on the intermediate stage. An intermediate classification error that the architecture is naturally robust to does not need to be corrected just because it looks imperfect in isolation on a per-stage classification report.

- **Decided NOT to retrain the Attack-Type NN to fix the Hulk/FTP confusion,** based on the above principle and the empirical 99.97% DQN compensation result. The classification report metric (NN accuracy) is not the deliverable — the remediation action accuracy is. The NN's confusion on this pair has zero measured impact on the deliverable.

- **Decided TO fix the Web_Brute_Force/Web_XSS confusion via action-level merging** rather than NN retraining, for two reasons: (1) no feature signal exists to train against — retraining would not help, (2) the two classes' differing optimal DQN actions meant there was no raw-feature fallback signal for the DQN to compensate with, unlike the Hulk/FTP case.

- **Action-merge safety reasoning:** Revoke Credentials is the more conservative choice for an ambiguous web-attack signal. If the true attack is Web_XSS (which Kill Process would have handled), revoking the attacker's credentials is still a meaningful and disruptive response with lower collateral damage risk than killing the wrong process.

**Updated results after Web attack merge retrain:**

| Metric | Before merge | After merge |
|---|---|---|
| Test action-match accuracy | 0.9901 | **0.9922** |
| Test average reward | 9.8552 | **9.8869** |
| Monitor-on-Attack rate | 0.0001 | 0.0001 (unchanged) |
| Web_Brute_Force accuracy | — | **99.51%** (408/410 → Revoke Credentials) |
| Web_XSS accuracy | — | **98.04%** (200/204 → Revoke Credentials) |

Both classes now achieve high accuracy against the same target action. The overall pipeline improvement (0.9901 → 0.9922) reflects the removal of the conflicting reward signal that the DQN previously received for Web_XSS rows.

**Next steps:**

- **Identified architecture gap:** the Attack-Type NN has no mechanism to express "I don't recognize this attack type." It always outputs a prediction across its 11 trained classes, even for flows from attack types it was never trained on (Heartbleed n=12, Web_SQL_Injection n=24 — both excluded from training due to insufficient samples) or any future zero-day attack type. In these cases the NN outputs an arbitrary softmax distribution, and the DQN acts on a meaningless signal.

- **Planned fix:** add a **confidence-threshold gate** between Stage 2 (Attack-Type NN) and Stage 3 (DQN Agent). If the NN's max-softmax confidence falls below a tunable threshold, route the flow to a "low-confidence / unknown attack" path rather than passing an uncertain classification to the DQN. This gate will eventually connect to Stage 4 (planned LLM-agent human-escalation layer) but can initially default to a safe "Monitor + alert" action. To be designed and built next.

---

### 📅 1 July 2026 — FastAPI Backend Scaffolding & Refactoring

**Topics covered:**
- Set up and refactored the complete FastAPI + PostgreSQL backend for the project
- Designed and enforced clean separation of concerns across the backend package structure
- Implemented a server-push WebSocket pattern and a multi-model loading system at startup

**Backend package structure finalised (as of today):**

```
backend/
├── main.py                         # FastAPI app entrypoint + lifespan startup
├── inference.py                    # inference logic stubs
├── schemas.py                      # Pydantic request/response models
├── __init__.py
│
├── db/                             # Database layer
│   ├── database.py                 # Engine, SessionLocal, check_db_connection()
│   ├── database_models.py          # SQLAlchemy ORM: Incident table + declarative Base
│   └── init_db.py                  # get_db() FastAPI dependency + init_db() startup helper
│
├── routers/                        # FastAPI route handlers
│   ├── health_route.py             # GET /health
│   ├── incident_route.py           # GET /incidents
│   ├── predict_route.py            # POST /predict
│   └── connection_route.py         # WS  /ws/connect
│
├── websocket/                      # WebSocket infrastructure
│   └── connection.py               # ConnectionManager class + module-level manager singleton
│
└── models/                         # ML model loader
    └── init_models.py              # load_models() — loads all 4 models at startup
```

**Key refactors done today:**

| Change | Rationale |
|---|---|
| ORM models moved from `database.py` → `database_models.py` | Models file owns the schema; avoids circular imports with `Base` |
| `get_db()` + `init_db()` moved to `db/init_db.py` | Separates DB connection config from app-level helpers |
| `ConnectionManager` moved from `routers/websocket.py` → `websocket/connection.py` | Class lives in its own module; `manager` singleton importable everywhere |
| All routers renamed to `*_route.py` convention | `health_route.py`, `incident_route.py`, `predict_route.py`, `connection_route.py` |
| WebSocket route `/ws/alerts` → `/ws/connect` | Semantically cleaner — connect endpoint, not an alerts source |

**WebSocket pattern implemented (server-push only):**
- Client connects to `ws://localhost:8000/ws/connect` and just listens
- Server voluntarily pushes JSON alert payloads whenever `POST /predict` detects an anomaly
- Client never needs to send anything — `manager.broadcast(payload)` called from `predict_route.py`
- Disconnect detected via `WebSocketDisconnect` exception in a background `receive()` loop

**ML model loader (`backend/models/init_models.py`):**
- Loads all 4 models once at startup via `load_models()` — into module-level variables, never reloaded
- Each model loads independently — missing file logs a warning, does not crash the server

| Variable | File | Type |
|---|---|---|
| `autoencoder` | `models/autoencoder.pt` | PyTorch |
| `hybrid_classifier` | `models/hybrid_classifier.pkl` | scikit-learn / joblib |
| `attack_type_nn` | `models/attack_type_nn.pt` | PyTorch |
| `dqn_agent` | `models/dqn_agent.pt` | PyTorch |

Supporting artefacts also loaded at startup: `scaler.pkl`, `hybrid_label_encoder.pkl`, `attack_type_label_map.json`, `threshold.json`.

**Python concepts reinforced:**
- `__init__.py` required even in Python 3.3+ for uvicorn/pytest to resolve packages correctly from project root
- `from __future__ import annotations` defers type-hint evaluation — no runtime cost, avoids forward-reference errors
- Module-level singletons (e.g. `manager = ConnectionManager()`) are cached by Python's import system — every import gets the same object in memory, so WebSocket state is shared across all routes

**API endpoints as of today:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | DB connectivity + model readiness check |
| `GET` | `/incidents` | Paginated stored incident list |
| `POST` | `/predict` | Submit a network flow for inference |
| `WS` | `/ws/connect` | Server-push alert subscription |

**Files created/renamed today:**
- `backend/db/database_models.py` — ORM models (renamed from `models.py`)
- `backend/db/init_db.py` — `get_db()` + `init_db()` (extracted from `database.py`)
- `backend/websocket/connection.py` — `ConnectionManager` class
- `backend/models/init_models.py` — 4-model startup loader
- `backend/routers/health_route.py` — renamed from `health.py`
- `backend/routers/incident_route.py` — renamed from `incidents.py`
- `backend/routers/predict_route.py` — renamed from `predict.py`
- `backend/routers/connection_route.py` — renamed from `websocket.py`

---

### 📅 2 July 2026 — Full Pipeline Integration, Model Loading Fix & Frontend Live Feed

**Context:** First end-to-end integration test of the complete system (simulator → FastAPI → WebSocket → React dashboard). Multiple bugs were identified and resolved to bring all four stages of the pipeline online simultaneously.

---

#### Backend — Model Loading Overhaul (`backend/models/init_models.py`)

The `load_models()` function was previously calling `torch.load()` expecting a full PyTorch model object. However, all `.pt` files in `models/` were saved using `torch.save(model.state_dict(), ...)` — i.e. they contain only the **state dictionary** (an `OrderedDict` of weights), not a full model instance. Calling `.eval()` on an `OrderedDict` crashed the startup loader, leaving all four `*_ready` flags as `False`.

**Fix:** Rewrote `load_models()` to use the correct **instantiate-then-load** pattern:
1. Import the model class definitions from the training scripts (`Autoencoder`, `AttackTypeNN`, `DQNNetwork`)
2. Instantiate each model with its correct constructor arguments (`input_dim`, `n_classes`, `state_dim`, etc.)
3. Call `model.load_state_dict(torch.load(..., weights_only=True))`
4. Call `model.eval()` on the live model object

The `n_classes` value for `AttackTypeNN` and `state_dim` for `DQNNetwork` are now derived at load-time from `attack_type_label_map.json` to remain self-consistent with training.

**Also fixed:** The `_ROOT` path constant used `parents[3]` (pointed to `E:\IISC\`, one level above the project root). Changed to `parents[2]` which correctly resolves to `E:\IISC\IISC_RESEARCH_INTERNSHIP\`.

---

#### Backend — Module Import Consolidation

`backend/main.py` and `backend/routers/health_route.py` were importing model-readiness flags from the old, unimplemented `backend/inference.py` placeholder instead of the active `backend/models/init_models.py`. This meant `load_models()` was never actually called on the correct module at startup.

**Fix:** Updated both files to `import backend.models.init_models as _inf` and reference `_inf.autoencoder_ready` / `_inf.dqn_agent_ready`.

---

#### Backend — Hybrid Label Encoder Fix (`backend/anomaly_classifier/anomaly_detection.py`)

The `hybrid_label_encoder.pkl` file produced by the training pipeline stores a plain Python **list** of class names (e.g. `['Benign', 'Botnet_ARES', 'DoS_Hulk', ...]`), not a `sklearn.LabelEncoder` object. The anomaly detection module was calling `.inverse_transform()` on this list, which crashed with `AttributeError: 'list' object has no attribute 'inverse_transform'`.

**Fix:** Updated `_run_hybrid_classifier()` to check `isinstance(_state.hybrid_label_encoder, list)` and use direct index lookup (`hybrid_label_encoder[pred_idx]`) when a list is detected, falling back to `.inverse_transform()` only for actual sklearn encoder objects.

---

#### Backend — Double-Scaling Bypass Preserved

The simulator (`simulator/streamlit_app.py`) sends features that are **already MinMax-scaled** (loaded from `X_attacks.npy` which was produced by `preprocess_attack.py`). The `scale_features()` helper functions inside `anomaly_detection.py` and `attack_identifier.py` were preserved but are NOT called at inference time, preventing double-scaling of the input. The functions remain available for future use when raw (unscaled) inputs need to be processed.

---

#### Frontend — Live Threat Feed Wired to WebSocket (`frontend/incident-dashboard/src/App.tsx`)

The "Live Threat Feed" table was previously rendering a static hardcoded `RECENT_ALERTS` array and not displaying any real WebSocket data, even though the `useWebSocket` hook was correctly receiving messages.

**Fixes applied:**
- Replaced `RECENT_ALERTS.map(...)` with `messages.slice().reverse().map(...)` to render live incoming WebSocket messages in reverse-chronological order (newest at top)
- Corrected the payload field mapping: backend broadcasts `dqn_action` (not `action`); the frontend now reads `msg.data?.dqn_action`
- Extended the table layout to show all key fields from each broadcast payload:

| Column | Source field |
|---|---|
| TIME | `msg.receivedAt` (browser timestamp) |
| CONNECTION | `source_ip` + `dest_ip` (two-line stacked) |
| ATTACK | `attack_type` (highlighted in amber) |
| ACTION | `dqn_action` + `recon_error` (sub-label) |
| SEV | Derived from action — `Block IP` → `critical`, `Isolate` → `warning`, else `info` |

- Added `maxHeight: 400px` + `overflow-y: auto` so the feed scrolls without pushing the rest of the layout
- Added `Listening for threats...` placeholder when no messages have arrived yet

---

#### Simulator — Garbled Text / Encoding Repair (`simulator/streamlit_app.py`)

The file had accumulated Mojibake across multiple sessions where UTF-8 bytes were mis-read as Latin-1 and then re-saved as UTF-8. Several emoji icons were also corrupted at the byte level beyond automatic recovery.

**Fix:** Rewrote `streamlit_app.py` from scratch using **Python Unicode escape sequences** for all non-ASCII characters (e.g. `\U0001f916` for 🤖, `\u25b6` for ▶) so the file is guaranteed to be pure 7-bit-safe ASCII with explicit Unicode escapes — immune to future editor or shell re-encoding accidents. All log message symbols that previously used Unicode (—, ✗, ►, ◈) were replaced with ASCII equivalents (`-`, `[X]`, `>`, `◈`) in the log box strings.

**Verified:** `0 lines with latin-range chars` and `Syntax OK` after rewrite.

---

#### Files Modified Today

| File | Change |
|---|---|
| `backend/models/init_models.py` | Rewrote `load_models()` to use instantiate-then-`load_state_dict()` pattern; fixed `_ROOT` path |
| `backend/main.py` | Changed startup import from `backend.inference` → `backend.models.init_models` |
| `backend/routers/health_route.py` | Changed readiness flag import from `backend.inference` → `backend.models.init_models`; fixed `dqn_ready` → `dqn_agent_ready` |
| `backend/anomaly_classifier/anomaly_detection.py` | Fixed `_run_hybrid_classifier()` to support list-type label encoder; preserved `_scale_features()` helper but bypassed it to prevent double-scaling |
| `backend/attacktype_classifier/attack_identifier.py` | Preserved `_scale_features()` helper but bypassed it in `identify_attack()` to avoid double-scaling of simulator features |
| `backend/dqn_agent/dqn_suggestion.py` | Verified state vector generation correctly passes raw (unscaled) features to the DQN per training design |
| `frontend/incident-dashboard/src/App.tsx` | Wired Live Threat Feed to live WebSocket `messages`; extended table columns; fixed `dqn_action` field mapping; added scroll and severity badge logic |
| `simulator/streamlit_app.py` | Full rewrite with Unicode escapes to permanently eliminate Mojibake |

---

### 📅 07 July 2026 — Frontend UI Refactoring, Live Dashboard Enhancements & Streaming Delay

**Topics covered:**
- Redesigned the primary application layout into a true multi-page architecture (`RootLayout`, `HomePage`, `IncidentsPage`, `TerminalPage`, `AnalyticsPage`).
- Enhanced the UI aesthetic with a floating, frosted-glass Navbar that dynamically reacts to scroll.
- Introduced a 200ms `asyncio.sleep(0.2)` inter-stage delay into the `predict_route.py` FastApi backend. This slows the processing just enough to visualize the pipeline steps streaming across the frontend terminal.
- Built a highly robust WebSocket log parser (`lib/logFormatter.ts`) that intercepts the raw JSON payloads and converts them into structured, color-coded, human-readable terminal lines (e.g. `15:52:26.123 [ANOMALY] 192.168.1.45 → 10.0.0.12 | recon_error=0.0721`).
- Upgraded the `TerminalLine` component to support an interactive UI: clicking on any formatted terminal log drops down a `<pre>` block displaying the original raw JSON payload.
- Integrated `attack_type_label_map.json` directly into the `IncidentsPage` frontend to properly decode numeric attack IDs back into readable names.

**Key achievements:**
- Transitioned the app from a single-page prototype to a scalable multi-page dashboard.
- Solved the "JSON dump" problem in the terminal view by successfully bridging the raw technical data with a beautiful, SOC-analyst-friendly UI, without losing access to the raw payload data.
- Stabilized the frontend WebSocket context manager to efficiently handle 500+ messages in memory using the new lightweight `FormattedLog` interface.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE TRAINING                             │
│                                                                      │
│  CICIDS2017 CSVs ──► Preprocessing ──► Autoencoder (v2, 24-dim)     │
│  UNSW-NB15 CSVs      (Pandas/NumPy)    (PyTorch+CUDA)               │
│                            │                  │                      │
│                       scaler.pkl         autoencoder.pt             │
│                       threshold.json     hybrid_classifier.pkl      │
│                       attack_type_nn.pt  dqn_agent.pt               │
└───────────────────────────┬──────────────────┘──────────────────────┘
                            │ load at startup
┌───────────────────────────▼──────────────────────────────────────────┐
│                         ONLINE PRODUCTION (4-Stage Pipeline)         │
│                                                                      │
│  Network Flow                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────┐                                │
│  │  Stage 1A: Autoencoder (v2)     │  Reconstruction error > T?     │
│  │  + Stage 1B: Hybrid Classifier  │  → YES → anomaly confirmed     │
│  └────────────────┬────────────────┘  → NO  → log benign, discard   │
│                   │ anomaly confirmed                                │
│                   ▼                                                  │
│  ┌─────────────────────────────────┐                                │
│  │  Stage 2: Attack-Type NN        │  Which of 13 CICIDS2017        │
│  │  (115→128→64→N, softmax)        │  attack types is this flow?    │
│  └────────────────┬────────────────┘  (+ confidence score)          │
│                   │ attack type + probs                              │
│                   ▼                                                  │
│  ┌─────────────────────────────────┐                                │
│  │  Stage 3: DQN Agent             │  State = [115 features |       │
│  │  (state_dim→128→64→5)           │    AE error | attack-type probs│
│  │                                 │    | confidence]               │
│  │  Actions: Block IP /            │  → select optimal action       │
│  │  Revoke Creds / Isolate Server /│                                │
│  │  Kill Process / Monitor         │                                │
│  └────────────────┬────────────────┘                                │
│                   │                                                  │
│                   ├── [high confidence] → Execute action directly   │
│                   │                                                  │
│                   └── [low confidence / future: Stage 4]            │
│                       LLM Agent Layer (planned — NOT built yet)      │
│                       → human-readable summary → analyst review      │
│                                                                      │
│  ──────────────────────────────────────────────────────────         │
│  WebSocket broadcast → React Dashboard ◄──── PostgreSQL             │
└──────────────────────────────────────────────────────────────────────┘
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
│       ├── X_attacks.npy        # all attack flows (600,141 × 115, MinMax-scaled)
│       ├── y_attacks.npy        # integer-encoded attack type labels
│       ├── y_attacks_str.npy    # string attack type labels
│       ├── scaler.pkl           # fitted MinMaxScaler (also in models/)
│       ├── feature_names.json   # ordered feature column list
│       ├── attack_label_map.json
│       ├── attack_type_label_map.json  # 11-class map for Attack-Type NN
│       ├── X_train_attacktype.npy      # attack-type NN training split
│       ├── X_val_attacktype.npy
│       ├── X_test_attacktype.npy
│       ├── y_train_attacktype.npy
│       ├── y_val_attacktype.npy
│       ├── y_test_attacktype.npy
│       ├── sample_weights_train_attacktype.npy  # balanced class weights
│       ├── dqn_states.npy               # full state vectors (115+1+N+1 dims)
│       ├── dqn_optimal_actions.npy      # ground-truth action labels (reward only)
│       ├── dqn_labels_str.npy           # string attack labels for per-class eval
│       ├── dqn_train_*.npy / dqn_val_*.npy / dqn_test_*.npy  # DQN splits
│       └── attack_class_counts.json
├── simulator/               # CICIDS2017 replay simulator (demo infrastructure)
│   ├── replay_simulator.py  # streams CSV rows to /predict as if live traffic
│   ├── streamlit_app.py     # interactive Streamlit UI to manually inject specific
│   │                        #   attack types into the backend for testing/debugging
│   └── README.md            # usage: --rate, --benign-ratio, --host flags
├── notebooks/               # Jupyter notebooks for EDA
├── data_preparation/             # Raw → processed data pipeline
│   ├── preprocess_benign.py      # Step 1 — benign data → autoencoder training data
│   ├── preprocess_attack.py      # Step 2 — attack data → DQN environment data
│   ├── build_hybrid_dataset.py   # Stage 1B data prep — 7-class hybrid dataset
│   └── build_attack_type_dataset.py  # Stage 2 data prep — 11-class attack-type dataset
├── training/
│   ├── train_autoencoder.py          # Stage 1A — autoencoder training
│   ├── train_hybrid_classifier.py    # Stage 1B training — XGBoost hybrid classifier
│   ├── train_attack_type_nn.py       # Stage 2 training — feedforward attack-type NN
│   ├── build_dqn_environment.py      # Stage 3 data prep — builds full DQN state vectors
│   │                                 #   (runs AE + Attack-Type NN forward passes)
│   ├── dqn_env.py                    # Stage 3 Gymnasium env — RemediationEnv (single-step)
│   └── train_dqn.py                  # Stage 3 training — DQN consumes [features | AE error
│                                     #   | attack-type NN softmax probs | confidence]
├── evaluation/                   # Model evaluation & diagnosis scripts
│   ├── evaluate_combined_pipeline.py
│   ├── eval_threshold.py
│   ├── analyze_feature_errors.py
│   ├── compare_models.py
│   ├── diagnose_dos_hulk.py
│   ├── diagnose_hulk_ftp_confusion.py
│   └── diagnose_nn_confusions.py
├── tests/
│   ├── test_autoencoder.py
│   ├── check_gpu.py
│   └── modelweight.py
├── models/                  # saved artefacts (output of training)
│   ├── autoencoder.pt          # ~90 KB  (state_dict, 115→80→48→24→48→80→115)
│   ├── hybrid_classifier.pkl   # ~2.5 MB (XGBoost, 7-class)
│   ├── hybrid_label_encoder.pkl
│   ├── attack_type_nn.pt       # feedforward NN (115→128→64→11)
│   ├── attack_type_label_map.json  # int → class-name mapping
│   ├── attack_type_nn_history.json # train/val loss per epoch
│   ├── dqn_agent.pt            # ~95 KB  (state_dim→128→64→5)
│   ├── dqn_training_history.json   # reward curve over episodes
│   ├── scaler.pkl              # ~7 KB   (copy from data/processed/)
│   ├── threshold.json          # <1 KB   (95th-percentile val MSE)
│   └── training_history.json   # autoencoder train/val loss per epoch
├── backend/
│   ├── main.py                     # FastAPI app entrypoint + lifespan startup
│   ├── inference.py                # inference logic stubs
│   ├── schemas.py                  # Pydantic request/response models
│   ├── __init__.py
│   ├── db/                         # Database layer
│   │   ├── database.py             # Engine, SessionLocal, check_db_connection()
│   │   ├── database_models.py      # SQLAlchemy ORM: Incident table + Base
│   │   └── init_db.py              # get_db() dependency + init_db() startup helper
│   ├── routers/                    # FastAPI route handlers
│   │   ├── health_route.py         # GET /health
│   │   ├── incident_route.py       # GET /incidents
│   │   ├── predict_route.py        # POST /predict
│   │   └── connection_route.py     # WS  /ws/connect
│   ├── websocket/                  # WebSocket infrastructure
│   │   └── connection.py           # ConnectionManager class + manager singleton
│   ├── models/                     # ML model loader
│   │   └── init_models.py          # load_models() — loads all 4 models at startup
│   ├── anomaly_classifier/
│   │   └── anomaly_detection.py
│   ├── attacktype_classifier/
│   │   └── attack_identifier.py
│   └── dqn_agent/
│       └── dqn_suggestion.py
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

# 10. (Optional) Launch the interactive Streamlit simulator UI
#     Lets you manually select an attack type and inject a single real flow
#     into the backend — useful for testing and debugging specific attack classes
#     NOTE: install streamlit first if not already: pip install streamlit
streamlit run simulator/streamlit_app.py
```

---

*README last updated: 7 July 2026 — Implemented multi-page UI architecture with floating navbar, structured terminal log parsing with JSON expansion, and backend streaming delays.*