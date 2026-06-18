"""
replay_simulator.py — CICIDS2017 Network Flow Replay Simulator
===============================================================

PURPOSE:
  Simulates a live network traffic source by reading real flow records
  from CICIDS2017 CSV files and streaming them one at a time to the
  FastAPI /predict endpoint, as if they were arriving from a live
  network tap.

  Since the project does not have access to live network hardware
  (explicitly out of scope in the project charter: "Integration with
  real firewall or router hardware" and "Real-time packet capture
  (PCAP)" are excluded), this simulator bridges that gap for
  demonstration and validation purposes.

HOW IT WORKS:
  1. Startup  — load a fixed pool of rows from raw CICIDS2017 CSVs
               (configurable mix of benign + attack via --benign-ratio).
               Pool is shuffled once; no CSV re-reading per request.

  2. Main loop — every 1/rate seconds, pick one row at random and
                 POST its raw (unscaled) feature values to /predict.
                 Scaling is the backend's responsibility.

  3. Ground truth stays client-side — the true label is never sent to
     the backend. It is used locally to compute running precision/recall
     so you can sanity-check the pipeline during a live demo.

REQUEST SCHEMA (POST /predict):
  {
    "features":  [<115 floats, unscaled>],
    "src_ip":    "192.168.1.45",
    "dst_ip":    "192.168.0.1",
    "src_port":  12345,
    "dst_port":  80,
    "protocol":  6,
    "timestamp": "2026-06-18T17:08:19.123456"
  }

EXPECTED RESPONSE (from /predict):
  {
    "anomaly_score": 0.045,
    "is_anomaly":    true,
    "action":        "Block IP",
    "action_id":     0
  }

USAGE:
  python simulator/replay_simulator.py
  python simulator/replay_simulator.py --rate 2 --benign-ratio 0.7
  python simulator/replay_simulator.py --host http://localhost:8000 \\
                                       --pool-size 3000 --max-flows 500

CLI FLAGS:
  --host          Backend base URL          (default: http://localhost:8000)
  --endpoint      Predict endpoint path     (default: /predict)
  --rate          Flows per second          (default: 1.0)
  --benign-ratio  Fraction of pool benign   (default: 0.8)
  --pool-size     Total rows at startup     (default: 3000)
  --max-flows     Stop after N flows (-1=∞) (default: -1)
  --seed          Random seed               (default: 42)
  --stats-every   Print stats every N flows (default: 10)

NEXT STEP:
  Implement backend/main.py with a POST /predict endpoint that accepts
  this request schema and returns the expected response.
"""

import os
import sys
import json
import glob
import random
import time
import logging
import argparse
import datetime
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────
# Path configuration
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../simulator/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

BENIGN_DATA_DIR   = os.path.join(_PROJECT_ROOT, "data", "cicids2017", "benign_data")
ATTACK_DATA_DIR   = os.path.join(_PROJECT_ROOT, "data", "cicids2017", "attack_data")
PROCESSED_DIR     = os.path.join(_PROJECT_ROOT, "data", "processed")
FEATURE_NAMES_PATH = os.path.join(PROCESSED_DIR, "feature_names.json")
LOG_PATH          = os.path.join(_PROJECT_ROOT, "data", "processed", "simulator.log")

# ─────────────────────────────────────────────
# Metadata column candidates
# (CICIDS2017 CSVs may use lowercase or mixed-case column names)
# ─────────────────────────────────────────────
# The simulator looks for these in order and takes the first match found.
_META_CANDIDATES = {
    "src_ip":   ["src_ip", "Source IP", "source_ip", "Src IP"],
    "dst_ip":   ["dst_ip", "Destination IP", "destination_ip", "Dst IP"],
    "src_port": ["src_port", "Source Port", "source_port", "Src Port"],
    "dst_port": ["dst_port", "Destination Port", "destination_port", "Dst Port"],
    "protocol": ["protocol", "Protocol"],
    "label":    ["label", "Label"],
}

# Columns whose absence from the CSV is acceptable (will use placeholder)
_OPTIONAL_META = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol"}

# ─────────────────────────────────────────────
# Logging setup  (console + file)
# ─────────────────────────────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)

_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(),
    mode="w", encoding="utf-8", closefd=False, buffering=1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load feature names produced by preprocess_benign.py
# ═══════════════════════════════════════════════════════════
def load_feature_names() -> list:
    """
    Load the ordered list of 115 feature column names from
    data/processed/feature_names.json.

    This file is produced by preprocess_benign.py and defines
    exactly which columns are features (in what order) for the model.
    Run preprocess_benign.py first if this file doesn't exist.

    Returns
    -------
    list of str : 115 feature column names
    """
    if not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(
            f"feature_names.json not found at '{FEATURE_NAMES_PATH}'.\n"
            f"Run 'python training/preprocess_benign.py' first."
        )
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    log.info(f"  Loaded {len(names)} feature names from feature_names.json")
    return names


# ═══════════════════════════════════════════════════════════
# STEP 2 — Resolve metadata column names
# ═══════════════════════════════════════════════════════════
def _resolve_meta_cols(df_cols: list) -> dict:
    """
    Map canonical metadata key names to actual column names present in
    the DataFrame. Returns only the keys that could be resolved.

    Parameters
    ----------
    df_cols : list  — column names from a loaded DataFrame

    Returns
    -------
    dict  { canonical_key: actual_column_name, ... }
    """
    cols_set = set(df_cols)
    resolved = {}
    for key, candidates in _META_CANDIDATES.items():
        for candidate in candidates:
            if candidate in cols_set:
                resolved[key] = candidate
                break
    return resolved


# ═══════════════════════════════════════════════════════════
# STEP 3 — Load one CSV file and extract rows
# ═══════════════════════════════════════════════════════════
def _load_csv_rows(filepath: str, feature_names: list, is_benign: bool) -> list:
    """
    Load a single CICIDS2017 CSV file and return a list of row dicts.

    Each row dict has:
      - "features"  : list of 115 raw (unscaled) float values
      - "src_ip"    : str   (or "0.0.0.0" if column absent)
      - "dst_ip"    : str   (or "0.0.0.0" if column absent)
      - "src_port"  : int   (or 0 if column absent)
      - "dst_port"  : int   (or 0 if column absent)
      - "protocol"  : int   (or 0 if column absent)
      - "true_label": str   ("BENIGN" or attack type string)
      - "is_attack" : bool

    Parameters
    ----------
    filepath     : str   absolute path to a CICIDS2017 CSV file
    feature_names: list  115 feature column names (from feature_names.json)
    is_benign    : bool  True → treat all rows as benign
    """
    df = pd.read_csv(filepath, low_memory=False)

    # CICIDS2017-specific: strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Resolve which actual column names are present
    meta = _resolve_meta_cols(df.columns.tolist())

    # Keep only columns we need (features + any resolved metadata)
    feature_cols_present = [c for c in feature_names if c in df.columns]
    missing_features = [c for c in feature_names if c not in df.columns]
    if missing_features:
        log.warning(f"    {os.path.basename(filepath)}: {len(missing_features)} feature cols "
                    f"not found in CSV (will be set to 0.0): {missing_features[:3]}...")

    # Replace inf/-inf with NaN, drop NaN rows
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna(subset=feature_cols_present)

    rows = []
    for _, row in df.iterrows():
        # ── Extract feature values ──
        features = []
        for col in feature_names:
            if col in df.columns:
                try:
                    features.append(float(row[col]))
                except (ValueError, TypeError):
                    features.append(0.0)
            else:
                features.append(0.0)   # missing feature → zero

        # ── Extract metadata (never sent to backend) ──
        src_ip   = str(row[meta["src_ip"]])   if "src_ip"   in meta else "0.0.0.0"
        dst_ip   = str(row[meta["dst_ip"]])   if "dst_ip"   in meta else "0.0.0.0"
        src_port = int(row[meta["src_port"]]) if "src_port" in meta else 0
        dst_port = int(row[meta["dst_port"]]) if "dst_port" in meta else 0
        protocol = int(float(row[meta["protocol"]])) if "protocol" in meta else 0

        # ── Ground-truth label (client-side only) ──
        if is_benign:
            true_label = "BENIGN"
        else:
            if "label" in meta:
                raw_label = str(row[meta["label"]]).strip()
                true_label = raw_label if raw_label.upper() != "BENIGN" else "BENIGN"
            else:
                true_label = "UNKNOWN"

        rows.append({
            "features":   features,
            "src_ip":     src_ip,
            "dst_ip":     dst_ip,
            "src_port":   src_port,
            "dst_port":   dst_port,
            "protocol":   protocol,
            "true_label": true_label,
            "is_attack":  true_label.upper() != "BENIGN",
        })

    return rows


# ═══════════════════════════════════════════════════════════
# STEP 4 — Build the full row pool at startup
# ═══════════════════════════════════════════════════════════
def build_pool(
    feature_names: list,
    pool_size: int,
    benign_ratio: float,
    seed: int,
) -> list:
    """
    Load and sample a fixed pool of rows from raw CICIDS2017 CSVs.

    The pool is loaded once at startup to avoid re-reading 1.8 GB of
    CSVs on every request. The benign/attack mix is controlled by
    benign_ratio. The pool is shuffled with the given seed.

    Parameters
    ----------
    feature_names : list  115 feature column names
    pool_size     : int   total number of rows to sample
    benign_ratio  : float fraction of pool that is benign (0.0–1.0)
    seed          : int   random seed for reproducible sampling

    Returns
    -------
    list of row dicts (see _load_csv_rows for schema)
    """
    rng = random.Random(seed)

    n_benign = int(pool_size * benign_ratio)
    n_attack = pool_size - n_benign

    log.info(f"  Pool target  : {pool_size} rows  "
             f"({n_benign} benign + {n_attack} attack)")

    # ── Load benign rows ──────────────────────────────────
    benign_rows = []
    benign_files = sorted(glob.glob(os.path.join(BENIGN_DATA_DIR, "*.csv")))
    if not benign_files:
        log.warning(f"  No benign CSV files found in '{BENIGN_DATA_DIR}'")
    else:
        log.info(f"  Found {len(benign_files)} benign CSV file(s)")
        for fpath in benign_files:
            fname = os.path.basename(fpath)
            rows = _load_csv_rows(fpath, feature_names, is_benign=True)
            benign_rows.extend(rows)
            log.info(f"    {fname:40s}  loaded {len(rows):>7,} rows")

    # ── Load attack rows ──────────────────────────────────
    attack_rows = []
    attack_files = sorted(glob.glob(os.path.join(ATTACK_DATA_DIR, "*.csv")))
    if not attack_files:
        log.warning(f"  No attack CSV files found in '{ATTACK_DATA_DIR}'")
    else:
        log.info(f"  Found {len(attack_files)} attack CSV file(s)")
        for fpath in attack_files:
            fname = os.path.basename(fpath)
            rows = _load_csv_rows(fpath, feature_names, is_benign=False)
            attack_rows.extend(rows)
            log.info(f"    {fname:40s}  loaded {len(rows):>7,} rows")

    # ── Sample down to target sizes ───────────────────────
    if len(benign_rows) > n_benign:
        benign_sample = rng.sample(benign_rows, n_benign)
    else:
        log.warning(f"  Only {len(benign_rows):,} benign rows available "
                    f"(requested {n_benign:,}) — using all")
        benign_sample = benign_rows

    if len(attack_rows) > n_attack:
        attack_sample = rng.sample(attack_rows, n_attack)
    else:
        log.warning(f"  Only {len(attack_rows):,} attack rows available "
                    f"(requested {n_attack:,}) — using all")
        attack_sample = attack_rows

    pool = benign_sample + attack_sample
    rng.shuffle(pool)

    actual_benign = sum(1 for r in pool if not r["is_attack"])
    actual_attack = sum(1 for r in pool if r["is_attack"])
    log.info(f"\n  Pool built   : {len(pool):,} rows  "
             f"({actual_benign:,} benign  +  {actual_attack:,} attack)")

    return pool


# ═══════════════════════════════════════════════════════════
# STEP 5 — Build the POST /predict request payload
# ═══════════════════════════════════════════════════════════
def build_payload(row: dict) -> dict:
    """
    Build the JSON payload for POST /predict from a pool row.

    Features are sent RAW (unscaled) — the backend applies scaler.pkl
    before passing them to the autoencoder, matching the architecture
    spec. The true_label and is_attack fields are NOT included.

    Parameters
    ----------
    row : dict — one row from the pool (see _load_csv_rows for schema)

    Returns
    -------
    dict — JSON-serialisable request body
    """
    return {
        "features":  row["features"],
        "src_ip":    row["src_ip"],
        "dst_ip":    row["dst_ip"],
        "src_port":  row["src_port"],
        "dst_port":  row["dst_port"],
        "protocol":  row["protocol"],
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# STEP 6 — Send one flow to the backend
# ═══════════════════════════════════════════════════════════
def send_flow(
    session: requests.Session,
    url: str,
    row: dict,
    timeout: float = 5.0,
) -> Optional[dict]:
    """
    POST one flow to the backend /predict endpoint.

    Returns the parsed JSON response dict, or None if the request
    failed (connection error, timeout, or non-2xx response).
    Errors are logged as warnings — the simulator continues regardless.

    Parameters
    ----------
    session : requests.Session  (shared across calls for keep-alive)
    url     : str               full URL of the /predict endpoint
    row     : dict              one pool row (features + metadata)
    timeout : float             request timeout in seconds

    Returns
    -------
    dict | None
    """
    payload = build_payload(row)
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.warning("  ⚠  Backend not reachable — is 'uvicorn backend.main:app' running?")
    except requests.exceptions.Timeout:
        log.warning(f"  ⚠  Request timed out after {timeout}s")
    except requests.exceptions.HTTPError as e:
        log.warning(f"  ⚠  HTTP {e.response.status_code}: {e.response.text[:120]}")
    except Exception as e:
        log.warning(f"  ⚠  Unexpected error: {e}")
    return None


# ═══════════════════════════════════════════════════════════
# STEP 7 — Running statistics tracker
# ═══════════════════════════════════════════════════════════
class StatsTracker:
    """
    Tracks running precision, recall, and F1 by comparing the backend's
    is_anomaly prediction against the ground-truth label kept client-side.

    This is a sanity check tool for demo sessions — it does NOT modify
    any model behaviour.
    """

    def __init__(self):
        self.tp = 0   # attack correctly flagged as anomaly
        self.fp = 0   # benign incorrectly flagged as anomaly
        self.tn = 0   # benign correctly passed as normal
        self.fn = 0   # attack missed (passed as normal)
        self.errors = 0        # backend returned no response
        self.total_sent = 0
        self.action_counts: dict = defaultdict(int)

    def update(self, row: dict, response: Optional[dict]):
        """Record one prediction result."""
        self.total_sent += 1
        if response is None:
            self.errors += 1
            return

        predicted_attack = bool(response.get("is_anomaly", False))
        actually_attack  = row["is_attack"]

        if predicted_attack and actually_attack:
            self.tp += 1
        elif predicted_attack and not actually_attack:
            self.fp += 1
        elif not predicted_attack and not actually_attack:
            self.tn += 1
        else:
            self.fn += 1

        action = response.get("action", "—")
        self.action_counts[action] += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def print_stats(self, flow_num: int):
        """Print a formatted stats snapshot to the log."""
        log.info(
            f"\n  ── Stats @ flow {flow_num} ──────────────────────────────────────\n"
            f"  Sent      : {self.total_sent}  "
            f"(errors: {self.errors}, "
            f"responded: {self.total_sent - self.errors})\n"
            f"  TP={self.tp}  FP={self.fp}  TN={self.tn}  FN={self.fn}\n"
            f"  Precision : {self.precision:.3f}\n"
            f"  Recall    : {self.recall:.3f}\n"
            f"  F1        : {self.f1:.3f}\n"
            f"  Actions   : {dict(self.action_counts)}\n"
            f"  ─────────────────────────────────────────────────────────────"
        )


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CICIDS2017 replay simulator — streams flows to FastAPI /predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",         default="http://localhost:8000",
                        help="Backend base URL")
    parser.add_argument("--endpoint",     default="/predict",
                        help="Predict endpoint path")
    parser.add_argument("--rate",         type=float, default=1.0,
                        help="Flows per second to send")
    parser.add_argument("--benign-ratio", type=float, default=0.8,
                        help="Fraction of pool that is benign (0.0–1.0)")
    parser.add_argument("--pool-size",    type=int,   default=3000,
                        help="Total rows to load at startup")
    parser.add_argument("--max-flows",    type=int,   default=-1,
                        help="Stop after N flows (-1 = run until Ctrl+C)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for pool sampling and selection")
    parser.add_argument("--stats-every",  type=int,   default=10,
                        help="Print running stats every N flows")
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 65)
    log.info("CICIDS2017 REPLAY SIMULATOR")
    log.info("(Streams historical flows to FastAPI /predict as live traffic)")
    log.info("=" * 65)
    log.info(f"  Backend    : {args.host}{args.endpoint}")
    log.info(f"  Rate       : {args.rate} flow(s)/second  "
             f"(interval: {1.0/args.rate:.2f}s)")
    log.info(f"  Pool size  : {args.pool_size}  "
             f"(benign: {args.benign_ratio:.0%}  attack: {1-args.benign_ratio:.0%})")
    log.info(f"  Max flows  : {'∞' if args.max_flows < 0 else args.max_flows}")
    log.info(f"  Seed       : {args.seed}")

    # ── STEP 1: Load feature names ─────────────────────────
    log.info("\n[STEP 1] Loading feature names...")
    feature_names = load_feature_names()
    log.info(f"  {len(feature_names)} features confirmed")

    # ── STEP 2: Build pool ─────────────────────────────────
    log.info("\n[STEP 2] Building row pool from CICIDS2017 CSVs...")
    pool = build_pool(
        feature_names=feature_names,
        pool_size=args.pool_size,
        benign_ratio=args.benign_ratio,
        seed=args.seed,
    )

    if not pool:
        log.error("  Pool is empty — check that CICIDS2017 CSV files exist in:")
        log.error(f"    {BENIGN_DATA_DIR}")
        log.error(f"    {ATTACK_DATA_DIR}")
        sys.exit(1)

    # ── STEP 3: Prepare HTTP session ───────────────────────
    predict_url = args.host.rstrip("/") + args.endpoint
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    rng         = random.Random(args.seed + 1)   # separate rng for selection
    stats       = StatsTracker()
    interval    = 1.0 / max(args.rate, 0.01)

    log.info(f"\n[STEP 3] Starting replay loop  →  {predict_url}")
    log.info("  Press Ctrl+C to stop.\n")

    flow_num = 0
    try:
        while True:
            if args.max_flows >= 0 and flow_num >= args.max_flows:
                log.info(f"\n  Reached --max-flows={args.max_flows}. Stopping.")
                break

            flow_num += 1
            t_start = time.perf_counter()

            # Pick a row at random from the pool (with replacement)
            row = rng.choice(pool)

            # Send to backend
            response = send_flow(session, predict_url, row, timeout=5.0)

            # Update stats
            stats.update(row, response)

            # Log this flow
            true_str  = "ATTACK" if row["is_attack"] else "benign"
            pred_str  = "ANOMALY" if (response and response.get("is_anomaly")) else "normal"
            action    = response.get("action", "—") if response else "no response"
            score_str = f"{response.get('anomaly_score', 0):.4f}" if response else "—"
            correct   = "✅" if response and (
                bool(response.get("is_anomaly")) == row["is_attack"]
            ) else ("❌" if response else "⚠")

            log.info(
                f"  [{flow_num:>5}] {correct}  "
                f"true={true_str:<8}  pred={pred_str:<8}  "
                f"score={score_str:<8}  action={action:<22}  "
                f"src={row['src_ip']}"
            )

            # Print periodic stats summary
            if flow_num % args.stats_every == 0:
                stats.print_stats(flow_num)

            # Sleep for the remainder of the interval
            elapsed = time.perf_counter() - t_start
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("\n  Interrupted by user (Ctrl+C)")

    # ── Final stats ────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("REPLAY COMPLETE")
    stats.print_stats(flow_num)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
