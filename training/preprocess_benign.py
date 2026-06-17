"""
preprocess_benign.py — Benign Traffic Preprocessing Pipeline (Autoencoder Training Data)
==========================================================================================

PURPOSE:
  Processes ONLY the benign traffic files from data/cicids2017/benign_data/.
  The autoencoder is trained EXCLUSIVELY on benign (normal) traffic so that
  it learns to reconstruct normal patterns. Attack traffic produces a high
  reconstruction error, which becomes the anomaly signal.

INPUT FILES (data/cicids2017/benign_data/):
  - monday_benign.csv
  - tuesday_benign.csv
  - wednesday_benign.csv
  - thursday_benign.csv
  - friday_benign.csv

PREPROCESSING STEPS:
  1. Load and concatenate all benign CSV files
  2. Strip column name whitespace (CICIDS2017-specific gotcha)
  3. Drop unnamed index columns and non-feature identifier columns
  4. Replace inf / -inf with NaN, then drop NaN rows
  5. Remove exact duplicate rows
  6. Clip physics-impossible negative values to 0
  7. Encode the 'protocol' column (already numeric in CICIDS2017)
  8. Convert features to float32 (halves GPU memory vs float64)
  9. Fit MinMaxScaler on training split ONLY (no data leakage)
  10. Split: 70% train | 15% val | 15% test (all benign)
  11. Run sanity checks (NaN, Inf, value range)
  12. Save: X_train_benign.npy, X_val_benign.npy, X_test_benign.npy
           scaler.pkl (used by preprocess_attack.py + inference)
           preprocessing_benign_report.txt

OUTPUT (data/processed/):
  - X_train_benign.npy   — autoencoder training input (~70% of benign rows)
  - X_val_benign.npy     — autoencoder validation input (~15% of benign rows)
  - X_test_benign.npy    — autoencoder test input (~15% of benign rows)
  - scaler.pkl           — fitted MinMaxScaler (MUST be run BEFORE preprocess_attack.py)
  - feature_names.json   — ordered list of feature column names
  - preprocessing_benign_report.txt

IMPORTANT:
  Run this script BEFORE preprocess_attack.py.
  The attack preprocessor loads scaler.pkl produced here to apply the
  same normalization to attack data (no re-fitting — that would be data leakage).
"""

import os
import glob
import json
import pickle
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# Anchor paths to the project root regardless of where the script is run from:
#   python training/preprocess_benign.py    (from project root)
#   cd training && python preprocess_benign.py
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

BENIGN_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "cicids2017", "benign_data")
OUTPUT_DIR      = os.path.join(_PROJECT_ROOT, "data", "processed")
MODEL_DIR       = os.path.join(_PROJECT_ROOT, "models")
RANDOM_SEED     = 42

# Split ratios: 70% train | 15% val | 15% test
VAL_SIZE  = 0.15
TEST_SIZE = 0.15

# Non-feature identifier columns — carry no predictive value
DROP_COLS = ["flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"]

# Columns with physics-impossible negatives confirmed in CICIDS2017 scans
# (durations and counts must be >= 0)
CLIP_TO_ZERO_COLS = [
    "active_max", "active_mean", "active_min",
    "packet_IAT_min", "packet_IAT_max", "packet_IAT_total", "packets_IAT_mean",
    "bwd_packets_IAT_min", "bwd_packets_IAT_max",
    "bwd_packets_IAT_mean", "bwd_packets_IAT_total",
    "duration", "bwd_packets_rate", "packets_rate",
    # Also handle possible CICFlowMeter column name variants
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
]

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# utf-8 stream handler so the → arrow works on Windows cp1252 consoles
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
            os.path.join(OUTPUT_DIR, "preprocessing_benign.log"),
            mode="w", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load all benign CSV files
# ═══════════════════════════════════════════════════════════
def load_benign_files(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in the benign_data/ folder.
    Expected files: monday_benign.csv, tuesday_benign.csv, etc.
    All rows in these files should be benign traffic only.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            f"Expected: monday_benign.csv, tuesday_benign.csv, etc."
        )

    log.info(f"Found {len(csv_files)} CSV file(s) in '{data_dir}'")

    dfs = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath, low_memory=False)
        log.info(f"  Loaded {fname:45s}  →  {len(df):>9,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\nCombined benign dataset shape: {combined.shape}")
    return combined


# ═══════════════════════════════════════════════════════════
# STEP 2 — Fix column names (CICIDS2017-specific gotcha)
# ═══════════════════════════════════════════════════════════
def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017 columns have invisible leading/trailing spaces baked in.
    e.g. ' Label' instead of 'Label'. Strip all whitespace.
    Also drop any Unnamed index columns CICFlowMeter adds.
    """
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    log.info(f"\n[STEP 2] Column names stripped of whitespace.")
    log.info(f"         Total columns after fix: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 3 — Drop non-feature identifier columns
# ═══════════════════════════════════════════════════════════
def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are identifiers, not network flow features."""
    existing = [c for c in DROP_COLS if c in df.columns]
    # Also drop label columns if accidentally present in benign files
    label_variants = ["label", "Label", "label_enc", "is_attack", "class", "Class"]
    label_existing = [c for c in label_variants if c in df.columns]

    df = df.drop(columns=existing + label_existing, errors='ignore')
    log.info(f"\n[STEP 3] Dropped identifier columns  : {existing}")
    log.info(f"         Dropped label columns (if any): {label_existing}")
    log.info(f"         Shape after drop              : {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 4 — Remove inf / NaN values
# ═══════════════════════════════════════════════════════════
def remove_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017 contains inf and -inf from CICFlowMeter division-by-zero bugs.
    Affected: 'Flow Bytes/s', 'Flow Packets/s' (divide by zero when duration=0).
    NaN appears in std-deviation features when a flow has only one packet.
    A single inf in a batch makes the entire loss NaN → training dies immediately.
    """
    n_before = len(df)
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    n_nan = df.isnull().sum().sum()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    n_after = len(df)
    log.info(f"\n[STEP 4] Inf values found   : {n_inf:,}")
    log.info(f"         NaN values found   : {n_nan:,}")
    log.info(f"         Rows removed       : {n_before - n_after:,}  ({n_before:,} → {n_after:,})")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 5 — Remove duplicate rows
# ═══════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017 contains ~6% exact duplicate rows. Duplicates bias what
    the autoencoder thinks is 'normal' (some patterns over-represented).
    Must run AFTER step 4 (inf→NaN then dropna changes values, so dedup on clean data).
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_after  = len(df)
    log.info(f"\n[STEP 5] Duplicates removed: {n_before - n_after:,}  ({n_before:,} → {n_after:,} rows)")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 6 — Clip negative values to 0
# ═══════════════════════════════════════════════════════════
def clip_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip physics-impossible negatives to 0.
    Durations, packet counts, and inter-arrival times cannot be negative.
    """
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
# STEP 7 — Encode 'protocol' column + convert to float32
# ═══════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    - Encode 'protocol' if it's a string (in CICIDS2017 it's usually already numeric: 6, 17, 0)
    - Convert all features to float32 (PyTorch requirement; halves GPU memory vs float64)
    - Return as numpy array for scaler input.
    """
    if "protocol" in df.columns:
        try:
            df["protocol"] = df["protocol"].astype(float).astype(int)
            log.info(f"\n[STEP 7] 'protocol' already numeric — cast to int.")
        except (ValueError, TypeError):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["protocol"] = le.fit_transform(df["protocol"].astype(str))
            log.info(f"\n[STEP 7] 'protocol' label-encoded (was string).")

    # Check for remaining non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning(f"         Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    log.info(f"         Feature columns  : {len(df.columns)}")
    log.info(f"         Converting to float32...")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 8 — Train / Val / Test split (all benign)
# ═══════════════════════════════════════════════════════════
def split_benign(df: pd.DataFrame):
    """
    Split benign data into 70% train, 15% val, 15% test.
    No stratification needed (all rows are the same class: benign).
    The autoencoder uses X as BOTH input and target — no y labels needed.
    """
    # First split: train vs (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Second split: val vs test (equal halves of the 30% remainder)
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    log.info(f"\n[STEP 8] Train/Val/Test split (all benign):")
    log.info(f"         Train : {len(df_train):>9,} rows  ({len(df_train)/len(df)*100:.1f}%)")
    log.info(f"         Val   : {len(df_val):>9,} rows  ({len(df_val)/len(df)*100:.1f}%)")
    log.info(f"         Test  : {len(df_test):>9,} rows  ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════
# STEP 9 — Fit MinMaxScaler and normalize
# ═══════════════════════════════════════════════════════════
def fit_and_scale(X_train_df, X_val_df, X_test_df, output_dir: str, model_dir: str):
    """
    Fit MinMaxScaler ONLY on training data (prevents data leakage).
    Transform val and test sets with the same fitted scaler.

    Saves scaler.pkl to BOTH:
      - data/processed/scaler.pkl  (used by preprocess_attack.py)
      - models/scaler.pkl          (used by inference at runtime)

    MinMaxScaler chosen (vs StandardScaler) per README spec:
      → Normalizes all features to [0, 1] range
      → Required for autoencoder: sigmoid output spans [0,1]
      → Appropriate for skewed network flow distributions
    """
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train_df).astype(np.float32)
    X_val   = scaler.transform(X_val_df).astype(np.float32)
    X_test  = scaler.transform(X_test_df).astype(np.float32)

    # Save scaler to processed dir (for attack preprocessing)
    scaler_proc_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_proc_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save scaler to models dir (for inference at runtime)
    scaler_model_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_model_path, "wb") as f:
        pickle.dump(scaler, f)

    log.info(f"\n[STEP 9] MinMaxScaler fit on training set.")
    log.info(f"         Feature min (first 5): {scaler.data_min_[:5].round(4)}")
    log.info(f"         Feature max (first 5): {scaler.data_max_[:5].round(4)}")
    log.info(f"         Scaler saved → {scaler_proc_path}")
    log.info(f"         Scaler saved → {scaler_model_path}")

    return X_train, X_val, X_test, scaler


# ═══════════════════════════════════════════════════════════
# STEP 10 — Sanity checks
# ═══════════════════════════════════════════════════════════
def sanity_check(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """
    Critical checks before saving:
      - No NaN values (would make autoencoder loss NaN)
      - No Inf values (same reason)
      - Value range is [0.0, 1.0] (MinMaxScaler guarantee)
    """
    log.info(f"\n[STEP 10] Sanity checks:")
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        has_nan  = bool(np.isnan(arr).any())
        has_inf  = bool(np.isinf(arr).any())
        val_min  = float(arr.min())
        val_max  = float(arr.max())
        ok = "✅" if (not has_nan and not has_inf and val_min >= 0.0 and val_max <= 1.0) else "❌"
        log.info(
            f"  {ok} {name:15s}  shape={arr.shape}  "
            f"NaN={has_nan}  Inf={has_inf}  "
            f"range=[{val_min:.4f}, {val_max:.4f}]"
        )

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        raise ValueError("Training data contains NaN or Inf — preprocessing failed!")


# ═══════════════════════════════════════════════════════════
# STEP 11 — Save arrays and feature names
# ═══════════════════════════════════════════════════════════
def save_outputs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    output_dir: str,
):
    """Save processed numpy arrays and feature name list."""
    arrays = {
        "X_train_benign": X_train,
        "X_val_benign":   X_val,
        "X_test_benign":  X_test,
    }
    for name, arr in arrays.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        log.info(f"  Saved {name:20s}  shape={arr.shape}  dtype={arr.dtype}  →  {path}")

    # Save feature names so train_autoencoder.py and inference know column order
    feat_path = os.path.join(output_dir, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"  Saved feature_names.json  ({len(feature_names)} features)  →  {feat_path}")

    log.info(f"\n[STEP 11] All outputs saved to '{output_dir}'")


# ═══════════════════════════════════════════════════════════
# STEP 12 — Save preprocessing report
# ═══════════════════════════════════════════════════════════
def save_report(df_raw, df_clean, feature_names, X_train, X_val, X_test, output_dir: str):
    """Write a human-readable summary of the benign preprocessing run."""
    report_path = os.path.join(output_dir, "preprocessing_benign_report.txt")
    lines = [
        "=" * 65,
        "BENIGN DATA PREPROCESSING REPORT",
        "(Input for Autoencoder Training — CICIDS2017 Benign Files)",
        "=" * 65,
        "",
        f"Raw rows loaded      : {len(df_raw):,}",
        f"After inf/NaN drop   : {len(df_raw) - (len(df_raw) - len(df_clean)):,}  (approx; dedup follows)",
        f"After dedup          : {len(df_clean):,}",
        f"Rows removed (total) : {len(df_raw) - len(df_clean):,}",
        f"Feature columns      : {len(feature_names)}",
        "",
        "SPLIT SIZES:",
        f"  Train : {X_train.shape[0]:>9,} rows  ({X_train.shape[0]/len(df_clean)*100:.1f}%)",
        f"  Val   : {X_val.shape[0]:>9,} rows  ({X_val.shape[0]/len(df_clean)*100:.1f}%)",
        f"  Test  : {X_test.shape[0]:>9,} rows  ({X_test.shape[0]/len(df_clean)*100:.1f}%)",
        "",
        "PREPROCESSING STEPS APPLIED:",
        "  [STRIP]  Column name whitespace stripped (CICIDS2017 gotcha)",
        "  [DROP]   Identifier columns removed (flow_id, timestamp, src/dst ip)",
        "  [DROP]   Unnamed index columns removed",
        "  [INF]    inf / -inf replaced with NaN, then rows dropped",
        "  [DEDUP]  Exact duplicate rows removed",
        "  [CLIP]   Negative values clipped to 0 in duration/count columns",
        "  [FLOAT]  All features cast to float32",
        "  [SCALE]  MinMaxScaler fit on train split only → [0, 1] range",
        "",
        "OUTPUT FILES:",
        "  X_train_benign.npy   — autoencoder training data",
        "  X_val_benign.npy     — autoencoder validation data",
        "  X_test_benign.npy    — autoencoder test data",
        "  scaler.pkl           — fitted MinMaxScaler (used by preprocess_attack.py)",
        "  feature_names.json   — ordered feature column list",
        "",
        "NEXT STEP:",
        "  Run preprocess_attack.py to process attack data for the DQN agent.",
        "  That script loads scaler.pkl from this output to normalize attack flows.",
        "",
        "AUTOENCODER TRAINING NOTE:",
        "  The autoencoder uses X as BOTH input and reconstruction target.",
        "  No y labels are needed. TensorDataset(X_tensor, X_tensor).",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"\n[STEP 12] Report saved → {report_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("BENIGN DATA PREPROCESSING PIPELINE")
    log.info("(Produces training data for the Autoencoder)")
    log.info("=" * 65)
    log.info(f"  Input  : {BENIGN_DATA_DIR}")
    log.info(f"  Output : {OUTPUT_DIR}")
    log.info(f"  Models : {MODEL_DIR}")

    # 1. Load
    df_raw = load_benign_files(BENIGN_DATA_DIR)

    # 2. Fix column names
    df = fix_column_names(df_raw.copy())

    # 3. Drop identifiers and label columns
    df = drop_identifier_columns(df)

    # 4. Remove inf and NaN
    df = remove_inf_nan(df)

    # 5. Remove duplicates
    df = remove_duplicates(df)

    # 6. Clip negatives
    df = clip_negative_values(df)

    # 7. Prepare features (encode protocol, remove non-numeric)
    df = prepare_features(df)
    feature_names = df.columns.tolist()
    log.info(f"\n         Feature list ({len(feature_names)} columns): {feature_names[:5]} ...")

    # 8. Split (all benign, no stratification needed)
    df_train, df_val, df_test = split_benign(df)

    # 9. Fit scaler on train, transform all splits
    X_train, X_val, X_test, scaler = fit_and_scale(
        df_train, df_val, df_test, OUTPUT_DIR, MODEL_DIR
    )

    # 10. Sanity checks
    sanity_check(X_train, X_val, X_test)

    # 11. Save arrays
    save_outputs(X_train, X_val, X_test, feature_names, OUTPUT_DIR)

    # 12. Save report
    save_report(df_raw, df, feature_names, X_train, X_val, X_test, OUTPUT_DIR)

    log.info("\n✅  Benign preprocessing complete!")
    log.info(f"    Feature dimensions  : {X_train.shape[1]}")
    log.info(f"    Train samples       : {X_train.shape[0]:,}")
    log.info(f"    Val   samples       : {X_val.shape[0]:,}")
    log.info(f"    Test  samples       : {X_test.shape[0]:,}")
    log.info(f"\n    → Now run: python training/preprocess_attack.py")


if __name__ == "__main__":
    main()
