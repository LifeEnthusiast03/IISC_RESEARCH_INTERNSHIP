"""
preprocess_attack.py — Attack Traffic Preprocessing Pipeline (DQN Agent Training Data)
========================================================================================

PURPOSE:
  Processes ONLY the attack traffic files from data/cicids2017/attack_data/.
  The DQN agent uses attack flow data as the environment state when learning
  its response policy (block IP, revoke credentials, isolate server, etc.).

  IMPORTANT: This script MUST be run AFTER preprocess_benign.py.
  It loads the scaler.pkl produced by preprocess_benign.py to apply the
  SAME normalization. Re-fitting on attack data would be data leakage —
  the scaler must only ever see benign training data.

INPUT FILES (data/cicids2017/attack_data/):
  - botnet_ares.csv
  - ddos_loit.csv
  - dos_golden_eye.csv
  - dos_hulk.csv
  - dos_slowhttptest.csv
  - dos_slowloris.csv
  - ftp_patator.csv
  - heartbleed.csv
  - portscan.csv
  - ssh_patator-new.csv
  - web_brute_force.csv
  - web_sql_injection.csv
  - web_xss.csv

  Note: ssh_patator-new.csv matches the filename in the data folder.

PREPROCESSING STEPS:
  1. Load and concatenate all attack CSV files
  2. Strip column name whitespace (CICIDS2017-specific gotcha)
  3. Drop unnamed index columns and non-feature identifier columns
  4. Replace inf / -inf with NaN, then drop NaN rows
  5. Remove exact duplicate rows
  6. Clip physics-impossible negative values to 0
  7. Encode the 'protocol' column
  8. Encode string labels → integers (e.g. 'DoS_Hulk' → 3)
  9. Load scaler.pkl from benign run — transform attack features (NO re-fitting)
  10. Run sanity checks (NaN, Inf, value range)
  11. Save: X_attacks.npy, y_attacks.npy, y_attacks_str.npy
           attack_label_map.json, preprocessing_attack_report.txt

OUTPUT (data/processed/):
  - X_attacks.npy          — normalized attack feature matrix (float32)
  - y_attacks.npy          — integer-encoded attack type labels
  - y_attacks_str.npy      — original string labels (for reporting)
  - attack_label_map.json  — {attack_name: integer_code} mapping
  - attack_class_counts.json — {attack_name: row_count} for DQN env design
  - preprocessing_attack_report.txt

DQN USAGE:
  The DQN simulation environment samples rows from X_attacks to create
  attack states. The agent receives the attack feature vector as its
  observation and must select the optimal response action:
    0 → Block IP
    1 → Revoke Credentials
    2 → Isolate Server
    3 → Kill Process
    4 → Monitor (no action)
  Rewards are +10 (threat neutralized), -3 (false alarm), -5 (service disrupted).
"""

import os
import glob
import json
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

ATTACK_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "cicids2017", "attack_data")
OUTPUT_DIR      = os.path.join(_PROJECT_ROOT, "data", "processed")
MODEL_DIR       = os.path.join(_PROJECT_ROOT, "models")
SCALER_PATH     = os.path.join(OUTPUT_DIR, "scaler.pkl")   # produced by preprocess_benign.py

# Non-feature identifier columns — carry no predictive value
DROP_COLS = ["flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]

# Columns with physics-impossible negatives (durations/counts must be >= 0)
CLIP_TO_ZERO_COLS = [
    "active_max", "active_mean", "active_min",
    "packet_IAT_min", "packet_IAT_max", "packet_IAT_total", "packets_IAT_mean",
    "bwd_packets_IAT_min", "bwd_packets_IAT_max",
    "bwd_packets_IAT_mean", "bwd_packets_IAT_total",
    "duration", "bwd_packets_rate", "packets_rate",
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
]

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(),
    mode='w', encoding='utf-8', closefd=False, buffering=1
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, "preprocessing_attack.log"),
            mode="w", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load all attack CSV files
# ═══════════════════════════════════════════════════════════
def load_attack_files(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in the attack_data/ folder.
    Logs per-file row counts and unique label values found.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            f"Expected attack files: botnet_ares.csv, ddos_loit.csv, dos_hulk.csv, etc."
        )

    log.info(f"Found {len(csv_files)} attack CSV file(s) in '{data_dir}'")

    dfs = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath, low_memory=False)

        # Quick peek at labels before stripping (strip will be done globally in step 2)
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        if label_col:
            labels = df[label_col].unique().tolist()
            log.info(f"  Loaded {fname:45s}  →  {len(df):>9,} rows  |  labels: {labels}")
        else:
            log.info(f"  Loaded {fname:45s}  →  {len(df):>9,} rows  |  (no label column detected)")

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\nCombined attack dataset shape: {combined.shape}")
    return combined


# ═══════════════════════════════════════════════════════════
# STEP 2 — Fix column names
# ═══════════════════════════════════════════════════════════
def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names; drop Unnamed index columns."""
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    log.info(f"\n[STEP 2] Column names stripped. Total columns: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 3 — Separate and drop identifier columns
# ═══════════════════════════════════════════════════════════
def separate_labels_and_drop_ids(df: pd.DataFrame):
    """
    Extract the 'label' column (attack type string) before dropping it.
    Drop non-feature identifier columns.
    Returns (df_features, y_str).
    """
    # Detect label column (case-insensitive, handles ' Label' after stripping)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError(
            "No 'label' column found in attack data. "
            f"Available columns: {df.columns.tolist()}"
        )

    y_str = df[label_col].astype(str).values
    log.info(f"\n[STEP 3] Label column found: '{label_col}'")
    log.info(f"         Unique attack labels: {sorted(set(y_str))}")

    # Drop identifier columns + the label column from features
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop + [label_col], errors='ignore')
    # Also drop any secondary label columns if present
    for extra in ["label_enc", "is_attack", "class", "Class"]:
        if extra in df.columns:
            df = df.drop(columns=[extra])

    log.info(f"         Dropped identifier columns: {existing_drop}")
    log.info(f"         Shape after drop           : {df.shape}")

    return df, y_str


# ═══════════════════════════════════════════════════════════
# STEP 4 — Remove inf / NaN values
# ═══════════════════════════════════════════════════════════
def remove_inf_nan(df: pd.DataFrame, y_str: np.ndarray):
    """
    Replace inf/-inf with NaN, drop resulting NaN rows.
    y_str must be filtered in sync with df rows.
    """
    n_before = len(df)
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    n_nan = df.isnull().sum().sum()

    # Build a boolean mask for rows to keep
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    keep_mask = df_clean.notna().all(axis=1)

    df_clean = df_clean[keep_mask]
    y_str_clean = y_str[keep_mask.values]

    n_after = len(df_clean)
    log.info(f"\n[STEP 4] Inf values found   : {n_inf:,}")
    log.info(f"         NaN values found   : {n_nan:,}")
    log.info(f"         Rows removed       : {n_before - n_after:,}  ({n_before:,} → {n_after:,})")

    return df_clean, y_str_clean


# ═══════════════════════════════════════════════════════════
# STEP 5 — Remove duplicate rows
# ═══════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame, y_str: np.ndarray):
    """Remove exact duplicate feature rows; keep y_str in sync."""
    n_before = len(df)
    df_with_label = df.copy()
    df_with_label["__label__"] = y_str
    df_with_label = df_with_label.drop_duplicates()
    y_str_clean = df_with_label["__label__"].values
    df_clean = df_with_label.drop(columns=["__label__"])
    n_after = len(df_clean)
    log.info(f"\n[STEP 5] Duplicates removed: {n_before - n_after:,}  ({n_before:,} → {n_after:,} rows)")
    return df_clean, y_str_clean


# ═══════════════════════════════════════════════════════════
# STEP 6 — Clip negative values to 0
# ═══════════════════════════════════════════════════════════
def clip_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physics-impossible negatives in duration/count columns to 0."""
    cols_present = [c for c in CLIP_TO_ZERO_COLS if c in df.columns]
    total_clipped = 0
    for col in cols_present:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            df[col] = df[col].clip(lower=0)
            log.info(f"  Clipped {n_neg:>7,} negatives in '{col}'")
            total_clipped += n_neg

    log.info(f"\n[STEP 6] Negative value clipping complete. Total cells clipped: {total_clipped:,}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 7 — Encode 'protocol' + remove non-numeric columns
# ═══════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode protocol if string; cast all features to float32."""
    if "protocol" in df.columns:
        try:
            df["protocol"] = df["protocol"].astype(float).astype(int)
            log.info(f"\n[STEP 7] 'protocol' already numeric — cast to int.")
        except (ValueError, TypeError):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["protocol"] = le.fit_transform(df["protocol"].astype(str))
            log.info(f"\n[STEP 7] 'protocol' label-encoded (was string).")

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning(f"         Dropping remaining non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    log.info(f"         Feature columns  : {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 8 — Encode string labels to integers
# ═══════════════════════════════════════════════════════════
def encode_labels(y_str: np.ndarray, output_dir: str):
    """
    Encode attack type strings to integers for DQN environment.
    e.g. 'DoS Hulk' → 3, 'PortScan' → 7

    Saves attack_label_map.json for reference in DQN training.
    Returns (y_int, label_encoder).
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)

    label_map = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    map_path  = os.path.join(output_dir, "attack_label_map.json")
    with open(map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    log.info(f"\n[STEP 8] Attack label encoding:")
    for cls, idx in sorted(label_map.items()):
        count = int((y_str == cls).sum())
        log.info(f"         {cls:35s} → {idx:2d}  ({count:>9,} rows)")
    log.info(f"         Label map saved → {map_path}")

    return y_int, le


# ═══════════════════════════════════════════════════════════
# STEP 9 — Apply scaler (NO re-fitting)
# ═══════════════════════════════════════════════════════════
def apply_scaler(df: pd.DataFrame, scaler_path: str, feature_names: list) -> np.ndarray:
    """
    Load the MinMaxScaler fitted on benign training data and apply it
    to attack features. We NEVER re-fit here — that would be data leakage.

    Also verifies that the attack feature columns match the benign feature columns.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at '{scaler_path}'.\n"
            f"Run preprocess_benign.py FIRST to generate the scaler."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    log.info(f"\n[STEP 9] Loaded scaler from: {scaler_path}")

    # Verify feature alignment — attack data must have same columns as benign
    if feature_names:
        missing = [f for f in feature_names if f not in df.columns]
        extra   = [f for f in df.columns if f not in feature_names]
        if missing:
            log.warning(f"         ⚠ Columns in benign but NOT in attack: {missing}")
        if extra:
            log.warning(f"         ⚠ Columns in attack but NOT in benign: {extra}")
        # Re-order / select only the features the scaler was fitted on
        available = [f for f in feature_names if f in df.columns]
        df = df[available]
        log.info(f"         Aligned to {len(available)} benign feature columns.")

    X_attacks = scaler.transform(df).astype(np.float32)
    log.info(f"         Normalized attack data  shape: {X_attacks.shape}")
    log.info(f"         Value range after scaling     : [{X_attacks.min():.4f}, {X_attacks.max():.4f}]")
    log.info(f"         (values slightly outside [0,1] are expected for attack data)")

    return X_attacks


# ═══════════════════════════════════════════════════════════
# STEP 10 — Sanity checks
# ═══════════════════════════════════════════════════════════
def sanity_check(X_attacks: np.ndarray, y_int: np.ndarray):
    """
    Verify no NaN or Inf values remain in attack features.
    Note: values slightly outside [0,1] are expected (attack patterns
    may exceed benign feature ranges — this is intentional and correct).
    """
    has_nan = bool(np.isnan(X_attacks).any())
    has_inf = bool(np.isinf(X_attacks).any())
    val_min = float(X_attacks.min())
    val_max = float(X_attacks.max())
    ok = "✅" if (not has_nan and not has_inf) else "❌"

    log.info(f"\n[STEP 10] Sanity checks:")
    log.info(
        f"  {ok} X_attacks  shape={X_attacks.shape}  "
        f"NaN={has_nan}  Inf={has_inf}  "
        f"range=[{val_min:.4f}, {val_max:.4f}]"
    )
    log.info(
        f"  ✅ y_attacks  shape={y_int.shape}  "
        f"unique classes={len(np.unique(y_int))}  "
        f"range=[{y_int.min()}, {y_int.max()}]"
    )

    if has_nan or has_inf:
        raise ValueError("Attack data contains NaN or Inf after preprocessing — check the pipeline!")


# ═══════════════════════════════════════════════════════════
# STEP 11 — Save outputs
# ═══════════════════════════════════════════════════════════
def save_outputs(
    X_attacks: np.ndarray,
    y_int: np.ndarray,
    y_str: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: str,
):
    """Save attack feature arrays, labels, and class count summary."""
    # Feature matrix
    x_path = os.path.join(output_dir, "X_attacks.npy")
    np.save(x_path, X_attacks)
    log.info(f"  Saved X_attacks.npy      shape={X_attacks.shape}  dtype={X_attacks.dtype}  →  {x_path}")

    # Integer labels
    y_path = os.path.join(output_dir, "y_attacks.npy")
    np.save(y_path, y_int)
    log.info(f"  Saved y_attacks.npy      shape={y_int.shape}   dtype={y_int.dtype}   →  {y_path}")

    # String labels (for easier debugging and reporting)
    ys_path = os.path.join(output_dir, "y_attacks_str.npy")
    np.save(ys_path, y_str)
    log.info(f"  Saved y_attacks_str.npy  shape={y_str.shape}   →  {ys_path}")

    # Per-class counts — useful for designing DQN reward shaping and sampling
    from collections import Counter
    counts = dict(Counter(y_str.tolist()))
    counts_sorted = dict(sorted(counts.items(), key=lambda x: -x[1]))
    counts_path = os.path.join(output_dir, "attack_class_counts.json")
    with open(counts_path, "w") as f:
        json.dump(counts_sorted, f, indent=2)
    log.info(f"  Saved attack_class_counts.json                  →  {counts_path}")

    log.info(f"\n[STEP 11] All outputs saved to '{output_dir}'")


# ═══════════════════════════════════════════════════════════
# STEP 12 — Save preprocessing report
# ═══════════════════════════════════════════════════════════
def save_report(
    df_raw,
    df_clean,
    X_attacks: np.ndarray,
    y_int: np.ndarray,
    y_str: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: str,
):
    """Write a human-readable summary of the attack preprocessing run."""
    report_path = os.path.join(output_dir, "preprocessing_attack_report.txt")
    from collections import Counter
    counts = Counter(y_str.tolist())

    lines = [
        "=" * 65,
        "ATTACK DATA PREPROCESSING REPORT",
        "(Input for DQN Agent Training Environment — CICIDS2017 Attack Files)",
        "=" * 65,
        "",
        f"Raw rows loaded      : {len(df_raw):,}",
        f"After cleaning       : {len(df_clean):,}",
        f"Rows removed (total) : {len(df_raw) - len(df_clean):,}",
        f"Feature columns      : {X_attacks.shape[1]}",
        f"Total attack samples : {X_attacks.shape[0]:,}",
        "",
        "ATTACK CLASS DISTRIBUTION (after cleaning):",
    ]
    for cls in sorted(counts, key=lambda c: -counts[c]):
        pct = counts[cls] / len(y_str) * 100
        lines.append(f"  {cls:35s}: {counts[cls]:>9,}  ({pct:5.2f}%)")

    lines += [
        "",
        "PREPROCESSING STEPS APPLIED:",
        "  [STRIP]  Column name whitespace stripped",
        "  [SEP]    Label column extracted before feature processing",
        "  [DROP]   Identifier columns removed",
        "  [INF]    inf / -inf removed, NaN rows dropped",
        "  [DEDUP]  Exact duplicate rows removed",
        "  [CLIP]   Negative values clipped to 0",
        "  [FLOAT]  Features cast to float32",
        "  [ENC]    Attack type strings → integer labels (LabelEncoder)",
        "  [SCALE]  MinMaxScaler LOADED (not re-fitted) from preprocess_benign output",
        "",
        "OUTPUT FILES:",
        "  X_attacks.npy          — normalized attack features (float32)",
        "  y_attacks.npy          — integer-encoded attack type labels",
        "  y_attacks_str.npy      — original string attack type labels",
        "  attack_label_map.json  — {attack_name: integer_code} mapping",
        "  attack_class_counts.json — {attack_name: row_count}",
        "",
        "DQN USAGE:",
        "  The DQN environment samples rows from X_attacks as 'states'.",
        "  The agent selects response actions and receives rewards.",
        "  NOTE: DQN trains in simulation — attack data informs the environment design,",
        "        but DQN training itself uses the Gymnasium environment loop.",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"\n[STEP 12] Report saved → {report_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("ATTACK DATA PREPROCESSING PIPELINE")
    log.info("(Produces environment states for DQN Agent Training)")
    log.info("=" * 65)
    log.info(f"  Input  : {ATTACK_DATA_DIR}")
    log.info(f"  Scaler : {SCALER_PATH}  (from preprocess_benign.py)")
    log.info(f"  Output : {OUTPUT_DIR}")

    # Guard: check scaler exists before doing expensive data loading
    if not os.path.exists(SCALER_PATH):
        log.error("=" * 65)
        log.error("ERROR: scaler.pkl not found!")
        log.error(f"Expected at: {SCALER_PATH}")
        log.error("Run preprocess_benign.py first, then re-run this script.")
        log.error("=" * 65)
        raise FileNotFoundError(f"scaler.pkl not found at '{SCALER_PATH}'")

    # Load feature names (produced by preprocess_benign.py)
    feat_names_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    feature_names = []
    if os.path.exists(feat_names_path):
        with open(feat_names_path) as f:
            feature_names = json.load(f)
        log.info(f"  Loaded {len(feature_names)} feature names from feature_names.json")
    else:
        log.warning("  feature_names.json not found — column alignment will be skipped.")

    # 1. Load
    df_raw = load_attack_files(ATTACK_DATA_DIR)

    # 2. Fix column names
    df = fix_column_names(df_raw.copy())

    # 3. Separate labels and drop identifier columns
    df, y_str = separate_labels_and_drop_ids(df)

    # 4. Remove inf and NaN (keep y_str in sync)
    df, y_str = remove_inf_nan(df, y_str)

    # 5. Remove duplicates (keep y_str in sync)
    df, y_str = remove_duplicates(df, y_str)

    # 6. Clip negatives
    df = clip_negative_values(df)

    # 7. Prepare features (encode protocol, remove non-numeric)
    df = prepare_features(df)

    # 8. Encode string labels → integers
    y_int, label_encoder = encode_labels(y_str, OUTPUT_DIR)

    # 9. Apply scaler (loaded from benign run — NO re-fitting)
    X_attacks = apply_scaler(df, SCALER_PATH, feature_names)

    # 10. Sanity checks
    sanity_check(X_attacks, y_int)

    # 11. Save arrays
    save_outputs(X_attacks, y_int, y_str, label_encoder, OUTPUT_DIR)

    # 12. Save report
    save_report(df_raw, df, X_attacks, y_int, y_str, label_encoder, OUTPUT_DIR)

    from collections import Counter
    counts = Counter(y_str.tolist())

    log.info("\n✅  Attack preprocessing complete!")
    log.info(f"    Feature dimensions   : {X_attacks.shape[1]}")
    log.info(f"    Total attack samples : {X_attacks.shape[0]:,}")
    log.info(f"    Attack types         : {len(counts)}")
    log.info(f"\n    Attack sample counts:")
    for cls in sorted(counts, key=lambda c: -counts[c]):
        log.info(f"      {cls:35s}: {counts[cls]:>9,}")
    log.info(f"\n    → Now run: python training/train_autoencoder.py")
    log.info(f"    → Then run: python training/train_dqn.py")


if __name__ == "__main__":
    main()
