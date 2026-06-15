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
Live network flow
       │
       ▼
┌─────────────────────┐
│  Stage 1            │   Reconstruction error = ‖x − x̂‖²
│  Autoencoder        │   Trained ONLY on benign traffic
│  (Anomaly Detector) │   → High error = attack detected
└─────────┬───────────┘
          │  error > threshold?
          ▼
┌─────────────────────┐
│  Stage 2            │   Selects from 5 actions:
│  DQN Agent          │   Block IP / Revoke creds /
│  (Auto-Remediation) │   Isolate server / Kill process / Monitor
└─────────┬───────────┘
          │
          ▼
   PostgreSQL + React Dashboard
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
│   ├── cicids2017/          # raw CICIDS CSV files
│   └── unsw_nb15/           # raw UNSW-NB15 CSV files
├── notebooks/               # Jupyter notebooks for EDA
├── training/
│   ├── preprocess.py        # data cleaning + normalization
│   ├── train_autoencoder.py # Stage 1 training
│   └── train_dqn.py         # Stage 2 training
├── models/                  # saved artefacts
│   ├── autoencoder.pt
│   ├── dqn_agent.pt
│   ├── scaler.pkl
│   └── threshold.json
├── backend/
│   ├── main.py              # FastAPI app + WebSocket
│   ├── inference.py         # model inference logic
│   ├── database.py          # SQLAlchemy + PostgreSQL
│   └── schemas.py           # Pydantic models
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── hooks/
│   └── package.json
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
pip install -r requirements.txt

# 3. Run data preprocessing
python training/preprocess.py

# 4. Train the autoencoder
python training/train_autoencoder.py

# 5. Train the DQN agent
python training/train_dqn.py

# 6. Start the backend
uvicorn backend.main:app --reload

# 7. Start the frontend
cd frontend && npm install && npm start
```

---

*README last updated: 15 June 2026*
*Next update due: 21 June 2026 (Data Readiness & Engineering milestone)*