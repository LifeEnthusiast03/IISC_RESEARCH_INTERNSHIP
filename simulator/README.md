# Replay Simulator — Usage Guide

## What This Does

`replay_simulator.py` reads real flow records from your CICIDS2017 CSV files and
streams them one at a time to your FastAPI `/predict` endpoint as if they were
arriving from a live network tap.

Since this project has no access to live hardware (out of scope per project charter),
the simulator bridges that gap for demonstration and validation.

---

## Prerequisites

1. **CICIDS2017 CSVs** must be present:
   - `data/cicids2017/benign_data/*.csv`
   - `data/cicids2017/attack_data/*.csv`

2. **`feature_names.json`** must exist in `data/processed/` — run preprocessing first:
   ```bash
   python training/preprocess_benign.py
   ```

3. **Backend must be running** before starting the simulator:
   ```bash
   uvicorn backend.main:app --reload
   ```

4. **`requests` library** must be installed:
   ```bash
   pip install requests
   ```

---

## Basic Usage

```bash
# Stream 1 flow per second, 80% benign / 20% attack (default settings)
python simulator/replay_simulator.py

# 2 flows per second, more attack traffic for a dramatic demo
python simulator/replay_simulator.py --rate 2 --benign-ratio 0.5

# Run for exactly 200 flows then stop
python simulator/replay_simulator.py --max-flows 200

# Point at a different host (e.g., deployed backend)
python simulator/replay_simulator.py --host http://192.168.1.100:8000

# Full custom run
python simulator/replay_simulator.py \
  --host http://localhost:8000 \
  --rate 0.5 \
  --benign-ratio 0.7 \
  --pool-size 5000 \
  --max-flows 1000 \
  --seed 123 \
  --stats-every 20
```

---

## All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `http://localhost:8000` | Backend base URL |
| `--endpoint` | `/predict` | Predict endpoint path |
| `--rate` | `1.0` | Flows to send per second |
| `--benign-ratio` | `0.8` | Fraction of pool that is benign (0.0–1.0) |
| `--pool-size` | `3000` | Total rows loaded at startup |
| `--max-flows` | `-1` | Stop after N flows (`-1` = run until Ctrl+C) |
| `--seed` | `42` | Random seed (reproducible runs) |
| `--stats-every` | `10` | Print running precision/recall every N flows |

---

## How It Works

```
Startup (runs once)
  │
  ├─ Load feature_names.json (115 feature column names)
  ├─ Read benign CSVs → sample N * benign_ratio rows
  ├─ Read attack CSVs → sample N * (1 - benign_ratio) rows
  └─ Shuffle into pool of N rows

Main loop (repeats every 1/rate seconds)
  │
  ├─ Pick one row at random from the pool (with replacement)
  ├─ Build JSON payload (115 raw/unscaled features + metadata)
  ├─ POST to /predict
  ├─ Compare response is_anomaly vs ground-truth label (client-side only)
  └─ Print result + update running precision/recall
```

---

## Request / Response Schema

**POST /predict request:**
```json
{
  "features":  [0.23, 0.01, ...],
  "src_ip":    "192.168.1.45",
  "dst_ip":    "192.168.0.1",
  "src_port":  54321,
  "dst_port":  80,
  "protocol":  6,
  "timestamp": "2026-06-18T17:08:19.123456"
}
```

> Features are sent **unscaled** — the backend applies `scaler.pkl` before
> passing them to the autoencoder.

**Expected /predict response:**
```json
{
  "anomaly_score": 0.0423,
  "is_anomaly":    true,
  "action":        "Block IP",
  "action_id":     0
}
```

---

## Console Output

Each flow prints one line:
```
[    1] ✅  true=ATTACK   pred=ANOMALY   score=0.0831   action=Block IP              src=192.168.1.54
[    2] ✅  true=benign   pred=normal    score=0.0019   action=—                     src=192.168.0.12
[    3] ❌  true=ATTACK   pred=normal    score=0.0021   action=—                     src=10.0.0.5
[   10] ── Stats @ flow 10 ──
        Precision: 0.900  Recall: 0.800  F1: 0.847
        Actions: {"Block IP": 3, "Monitor": 2}
```

- `✅` = correct prediction (TP or TN)
- `❌` = wrong prediction (FP or FN)
- `⚠` = backend did not respond

---

## Ground-Truth Note

The true label is **never sent to the backend**. It stays client-side
and is used only for the running precision/recall display. This means
the evaluation is fair — the model only sees the 115 feature values,
not the label.

---

## Stopping

Press `Ctrl+C` at any time. The simulator prints a final stats summary
and exits cleanly.

---

## Logs

A log file is written to:
```
data/processed/simulator.log
```
