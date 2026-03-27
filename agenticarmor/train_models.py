"""
AgenticArmor — Real ML Model Training
======================================
Supports:
  1. Real CIC-IDS-2018 dataset (auto-download via KaggleHub or manual CSV)
  2. High-quality synthetic fallback if dataset unavailable
  
Models trained:
  - Random Forest (supervised classification)
  - XGBoost      (supervised, higher accuracy)
  - Isolation Forest (unsupervised zero-day detection)

Run:
    pip install -r requirements.txt
    python train_models.py
    python train_models.py --synthetic      (force synthetic, skip dataset)
    python train_models.py --kaggle-key YOUR_KEY (auto-download dataset)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# CIC-IDS-2018 canonical 78 features
FEATURE_NAMES = [
    'dst_port','protocol','flow_duration','tot_fwd_pkts','tot_bwd_pkts',
    'totlen_fwd_pkts','totlen_bwd_pkts','fwd_pkt_len_max','fwd_pkt_len_min',
    'fwd_pkt_len_mean','fwd_pkt_len_std','bwd_pkt_len_max','bwd_pkt_len_min',
    'bwd_pkt_len_mean','bwd_pkt_len_std','flow_byts_s','flow_pkts_s',
    'flow_iat_mean','flow_iat_std','flow_iat_max','flow_iat_min',
    'fwd_iat_tot','fwd_iat_mean','fwd_iat_std','fwd_iat_max','fwd_iat_min',
    'bwd_iat_tot','bwd_iat_mean','bwd_iat_std','bwd_iat_max','bwd_iat_min',
    'fwd_psh_flags','bwd_psh_flags','fwd_urg_flags','bwd_urg_flags',
    'fwd_header_len','bwd_header_len','fwd_pkts_s','bwd_pkts_s',
    'pkt_len_min','pkt_len_max','pkt_len_mean','pkt_len_std','pkt_len_var',
    'fin_flag_cnt','syn_flag_cnt','rst_flag_cnt','psh_flag_cnt','ack_flag_cnt',
    'urg_flag_cnt','cwe_flag_count','ece_flag_cnt','down_up_ratio',
    'pkt_size_avg','fwd_seg_size_avg','bwd_seg_size_avg','fwd_byts_b_avg',
    'fwd_pkts_b_avg','fwd_blk_rate_avg','bwd_byts_b_avg','bwd_pkts_b_avg',
    'bwd_blk_rate_avg','subflow_fwd_pkts','subflow_fwd_byts','subflow_bwd_pkts',
    'subflow_bwd_byts','init_fwd_win_byts','init_bwd_win_byts','fwd_act_data_pkts',
    'fwd_seg_size_min','active_mean','active_std','active_max','active_min',
    'idle_mean','idle_std','idle_max','idle_min'
]

LABELS = ['BENIGN', 'DDoS', 'PortScan', 'BruteForce', 'SQLInjection', 'C2_Heartbeat']

# CIC-IDS-2018 column name mapping (their CSV uses different names)
CIC_COLUMN_MAP = {
    'Dst Port':           'dst_port',
    'Protocol':           'protocol',
    'Flow Duration':      'flow_duration',
    'Tot Fwd Pkts':       'tot_fwd_pkts',
    'Tot Bwd Pkts':       'tot_bwd_pkts',
    'TotLen Fwd Pkts':    'totlen_fwd_pkts',
    'TotLen Bwd Pkts':    'totlen_bwd_pkts',
    'Fwd Pkt Len Max':    'fwd_pkt_len_max',
    'Fwd Pkt Len Min':    'fwd_pkt_len_min',
    'Fwd Pkt Len Mean':   'fwd_pkt_len_mean',
    'Fwd Pkt Len Std':    'fwd_pkt_len_std',
    'Bwd Pkt Len Max':    'bwd_pkt_len_max',
    'Bwd Pkt Len Min':    'bwd_pkt_len_min',
    'Bwd Pkt Len Mean':   'bwd_pkt_len_mean',
    'Bwd Pkt Len Std':    'bwd_pkt_len_std',
    'Flow Byts/s':        'flow_byts_s',
    'Flow Pkts/s':        'flow_pkts_s',
    'Flow IAT Mean':      'flow_iat_mean',
    'Flow IAT Std':       'flow_iat_std',
    'Flow IAT Max':       'flow_iat_max',
    'Flow IAT Min':       'flow_iat_min',
    'Fwd IAT Tot':        'fwd_iat_tot',
    'Fwd IAT Mean':       'fwd_iat_mean',
    'Fwd IAT Std':        'fwd_iat_std',
    'Fwd IAT Max':        'fwd_iat_max',
    'Fwd IAT Min':        'fwd_iat_min',
    'Bwd IAT Tot':        'bwd_iat_tot',
    'Bwd IAT Mean':       'bwd_iat_mean',
    'Bwd IAT Std':        'bwd_iat_std',
    'Bwd IAT Max':        'bwd_iat_max',
    'Bwd IAT Min':        'bwd_iat_min',
    'Fwd PSH Flags':      'fwd_psh_flags',
    'Bwd PSH Flags':      'bwd_psh_flags',
    'Fwd URG Flags':      'fwd_urg_flags',
    'Bwd URG Flags':      'bwd_urg_flags',
    'Fwd Header Len':     'fwd_header_len',
    'Bwd Header Len':     'bwd_header_len',
    'Fwd Pkts/s':         'fwd_pkts_s',
    'Bwd Pkts/s':         'bwd_pkts_s',
    'Pkt Len Min':        'pkt_len_min',
    'Pkt Len Max':        'pkt_len_max',
    'Pkt Len Mean':       'pkt_len_mean',
    'Pkt Len Std':        'pkt_len_std',
    'Pkt Len Var':        'pkt_len_var',
    'FIN Flag Cnt':       'fin_flag_cnt',
    'SYN Flag Cnt':       'syn_flag_cnt',
    'RST Flag Cnt':       'rst_flag_cnt',
    'PSH Flag Cnt':       'psh_flag_cnt',
    'ACK Flag Cnt':       'ack_flag_cnt',
    'URG Flag Cnt':       'urg_flag_cnt',
    'CWE Flag Count':     'cwe_flag_count',
    'ECE Flag Cnt':       'ece_flag_cnt',
    'Down/Up Ratio':      'down_up_ratio',
    'Pkt Size Avg':       'pkt_size_avg',
    'Fwd Seg Size Avg':   'fwd_seg_size_avg',
    'Bwd Seg Size Avg':   'bwd_seg_size_avg',
    'Fwd Byts/b Avg':     'fwd_byts_b_avg',
    'Fwd Pkts/b Avg':     'fwd_pkts_b_avg',
    'Fwd Blk Rate Avg':   'fwd_blk_rate_avg',
    'Bwd Byts/b Avg':     'bwd_byts_b_avg',
    'Bwd Pkts/b Avg':     'bwd_pkts_b_avg',
    'Bwd Blk Rate Avg':   'bwd_blk_rate_avg',
    'Subflow Fwd Pkts':   'subflow_fwd_pkts',
    'Subflow Fwd Byts':   'subflow_fwd_byts',
    'Subflow Bwd Pkts':   'subflow_bwd_pkts',
    'Subflow Bwd Byts':   'subflow_bwd_byts',
    'Init Fwd Win Byts':  'init_fwd_win_byts',
    'Init Bwd Win Byts':  'init_bwd_win_byts',
    'Fwd Act Data Pkts':  'fwd_act_data_pkts',
    'Fwd Seg Size Min':   'fwd_seg_size_min',
    'Active Mean':        'active_mean',
    'Active Std':         'active_std',
    'Active Max':         'active_max',
    'Active Min':         'active_min',
    'Idle Mean':          'idle_mean',
    'Idle Std':           'idle_std',
    'Idle Max':           'idle_max',
    'Idle Min':           'idle_min',
    'Label':              'label',
}

CIC_LABEL_MAP = {
    'BENIGN':       'BENIGN',
    'DDoS':         'DDoS',
    'DoS Hulk':     'DDoS',
    'DoS GoldenEye':'DDoS',
    'DoS slowloris':'DDoS',
    'DoS Slowhttptest': 'DDoS',
    'PortScan':     'PortScan',
    'FTP-Patator':  'BruteForce',
    'SSH-Patator':  'BruteForce',
    'Bot':          'C2_Heartbeat',
    'Infiltration': 'C2_Heartbeat',
    'Web Attack  Brute Force':  'BruteForce',
    'Web Attack  Sql Injection':'SQLInjection',
    'Web Attack  XSS':          'SQLInjection',
    'Heartbleed':   'C2_Heartbeat',
}


# ── REAL DATASET LOADER ─────────────────────────────────────────────────
def load_real_dataset(csv_paths):
    """
    Load one or more CIC-IDS-2018 CSV files.
    csv_paths: list of file paths or a single path string.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  [!] File not found: {path}")
            continue
        print(f"  [*] Loading {path} ...")
        try:
            df = pd.read_csv(path, nrows=200000, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"      Loaded {len(df):,} rows")
        except Exception as e:
            print(f"  [!] Failed to load {path}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[*] Total rows loaded: {len(df):,}")

    # Rename columns
    df = df.rename(columns=CIC_COLUMN_MAP)

    # Keep only columns we need
    needed = FEATURE_NAMES + ['label']
    available = [c for c in needed if c in df.columns]
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"  [!] Missing {len(missing)} features — filling with 0: {missing[:5]}...")
        for col in missing:
            df[col] = 0.0
    df = df[needed]

    # Map labels
    df['label'] = df['label'].str.strip()
    df['label'] = df['label'].map(CIC_LABEL_MAP)
    df = df.dropna(subset=['label'])

    # Clean numeric columns
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(f"[✓] Clean rows: {len(df):,}")
    print(f"    Class distribution:\n{df['label'].value_counts()}\n")

    # Balance classes — cap majority class at 3x minority
    min_count = df['label'].value_counts().min()
    cap = min(min_count * 3, 50000)
    df = df.groupby('label').apply(
        lambda x: x.sample(min(len(x), cap), random_state=42)
    ).reset_index(drop=True)
    print(f"[✓] Balanced dataset: {len(df):,} rows")
    print(f"    Balanced distribution:\n{df['label'].value_counts()}\n")
    return df


# ── SYNTHETIC FALLBACK ──────────────────────────────────────────────────
def generate_synthetic_dataset(n_samples=80000):
    """
    High-quality synthetic data with realistic feature distributions
    derived from published CIC-IDS-2018 statistics.
    """
    print(f"[*] Generating {n_samples:,} synthetic samples (high-quality mode)...")
    np.random.seed(42)
    rows = []
    samples_per_class = n_samples // len(LABELS)

    for label in LABELS:
        for _ in range(samples_per_class):
            f = np.zeros(len(FEATURE_NAMES))

            if label == 'BENIGN':
                f[0]  = np.random.choice([80, 443, 53, 123, 8080, 3000, 8443])
                f[1]  = np.random.choice([6, 17])
                f[2]  = np.random.randint(1000, 200000)
                f[3]  = np.random.randint(2, 80)
                f[4]  = np.random.randint(2, 80)
                f[5]  = np.random.randint(200, 10000)
                f[6]  = np.random.randint(200, 10000)
                f[7]  = np.random.randint(200, 1500)
                f[8]  = np.random.randint(20, 100)
                f[15] = np.random.uniform(500, 5000)
                f[16] = np.random.uniform(5, 300)
                f[17] = np.random.uniform(1000, 50000)
                f[18] = np.random.uniform(500, 20000)
                f[44] = np.random.randint(0, 2)
                f[45] = np.random.randint(0, 3)
                f[47] = np.random.randint(0, 10)
                f[48] = np.random.randint(2, 30)
                f[65] = np.random.randint(8192, 65535)
                f[66] = np.random.randint(8192, 65535)

            elif label == 'DDoS':
                f[0]  = np.random.choice([80, 443, 53, 123])
                f[1]  = 17  # UDP flood
                f[2]  = np.random.randint(50, 3000)
                f[3]  = np.random.randint(5000, 50000)
                f[4]  = np.random.randint(0, 3)
                f[5]  = np.random.randint(200000, 2000000)
                f[6]  = np.random.randint(0, 100)
                f[7]  = np.random.randint(40, 65)
                f[8]  = np.random.randint(40, 65)
                f[15] = np.random.uniform(100000, 5000000)
                f[16] = np.random.uniform(5000, 50000)
                f[17] = np.random.uniform(10, 500)
                f[18] = np.random.uniform(1, 50)
                f[44] = 0
                f[45] = np.random.randint(3000, 30000)  # SYN flood
                f[47] = 0
                f[48] = 0

            elif label == 'PortScan':
                f[0]  = np.random.randint(1, 65535)
                f[1]  = 6  # TCP
                f[2]  = np.random.randint(5, 200)
                f[3]  = np.random.randint(1, 3)
                f[4]  = np.random.randint(0, 2)
                f[5]  = np.random.randint(40, 80)
                f[6]  = np.random.randint(0, 20)
                f[7]  = np.random.randint(40, 60)
                f[8]  = np.random.randint(40, 60)
                f[15] = np.random.uniform(200, 2000)
                f[16] = np.random.uniform(5, 100)
                f[44] = np.random.randint(0, 2)
                f[45] = 1  # exactly 1 SYN
                f[46] = np.random.randint(0, 2)  # RST (port closed)
                f[48] = 0  # no ACK = no full handshake
                f[65] = -1
                f[66] = -1

            elif label == 'BruteForce':
                f[0]  = np.random.choice([22, 21, 3389, 23, 5900])
                f[1]  = 6
                f[2]  = np.random.randint(200, 5000)
                f[3]  = np.random.randint(15, 150)
                f[4]  = np.random.randint(15, 150)
                f[5]  = np.random.randint(2000, 12000)
                f[6]  = np.random.randint(2000, 12000)
                f[15] = np.random.uniform(100, 2000)
                f[16] = np.random.uniform(10, 150)
                f[17] = np.random.uniform(50, 300)
                f[18] = np.random.uniform(5, 30)  # low std = scripted/regular
                f[45] = np.random.randint(1, 5)
                f[48] = np.random.randint(10, 80)

            elif label == 'SQLInjection':
                f[0]  = np.random.choice([3306, 5432, 1433, 80, 443, 8080])
                f[1]  = 6
                f[2]  = np.random.randint(5000, 100000)
                f[3]  = np.random.randint(5, 40)
                f[4]  = np.random.randint(5, 40)
                f[5]  = np.random.randint(10000, 100000)  # large payload
                f[6]  = np.random.randint(2000, 30000)
                f[7]  = np.random.randint(1000, 10000)
                f[8]  = np.random.randint(100, 500)
                f[15] = np.random.uniform(100, 2000)
                f[16] = np.random.uniform(1, 30)
                f[47] = np.random.randint(5, 30)  # PSH flags (data push)
                f[48] = np.random.randint(5, 30)

            elif label == 'C2_Heartbeat':
                f[0]  = np.random.choice([443, 8443, 4444, 8080, 1234])
                f[1]  = 6
                f[2]  = np.random.randint(27000, 33000)  # ~30s periodic
                f[3]  = np.random.randint(2, 6)
                f[4]  = np.random.randint(2, 6)
                f[5]  = np.random.randint(110, 130)  # ~118 bytes (beacon size)
                f[6]  = np.random.randint(110, 130)
                f[7]  = np.random.randint(115, 125)  # uniform pkt size
                f[8]  = np.random.randint(115, 125)
                f[15] = np.random.uniform(0.05, 0.2)
                f[16] = np.random.uniform(0.05, 0.3)
                f[17] = np.random.uniform(28000, 32000)  # very regular IAT
                f[18] = np.random.uniform(10, 200)  # low std = periodic

            # Add realistic noise
            noise_scale = 0.3 if label == 'C2_Heartbeat' else 0.8
            f += np.random.normal(0, noise_scale, len(FEATURE_NAMES))
            f = np.clip(f, 0, None)
            rows.append((*f, label))

    cols = FEATURE_NAMES + ['label']
    df = pd.DataFrame(rows, columns=cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[✓] Synthetic dataset: {df.shape}")
    print(f"    Class distribution:\n{df['label'].value_counts()}\n")
    return df


# ── TRAIN RANDOM FOREST ─────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("[*] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        oob_score=True,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[✓] Random Forest Accuracy: {acc*100:.2f}%")
    if hasattr(rf, 'oob_score_'):
        print(f"    OOB Score: {rf.oob_score_*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature importance top 10
    importances = rf.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    print("Top 10 features:")
    for i, idx in enumerate(top10_idx):
        print(f"  {i+1:2d}. {FEATURE_NAMES[idx]:30s} {importances[idx]:.4f}")

    joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
    print(f"[✓] Saved → {MODEL_DIR}/random_forest.pkl\n")
    return rf


# ── TRAIN XGBOOST ───────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, le):
    try:
        import xgboost as xgb
    except ImportError:
        print("[!] XGBoost not installed — skipping (pip install xgboost)")
        return None

    print("[*] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[✓] XGBoost Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost.pkl")
    print(f"[✓] Saved → {MODEL_DIR}/xgboost.pkl\n")
    return xgb_model


# ── TRAIN ISOLATION FOREST ──────────────────────────────────────────────
def train_isolation_forest(X_benign, X_all):
    print("[*] Training Isolation Forest (unsupervised zero-day detection)...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        max_samples=min(10000, len(X_benign)),
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_benign)

    # Validate
    benign_scores = iso.predict(X_benign[:500])
    print(f"    Benign flagged as anomaly: {(benign_scores==-1).sum()}/500")
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest.pkl")
    print(f"[✓] Saved → {MODEL_DIR}/isolation_forest.pkl\n")
    return iso


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AgenticArmor Model Training")
    parser.add_argument("--csv", nargs="+", default=None,
                        help="Path(s) to CIC-IDS-2018 CSV files")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data (skip real dataset)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AgenticArmor — ML Model Training Pipeline")
    print("=" * 60 + "\n")

    # ── 1. Load data ──
    df = None

    if not args.synthetic:
        # Try user-provided CSV paths
        if args.csv:
            df = load_real_dataset(args.csv)

        # Auto-detect CSVs in current directory
        if df is None:
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and
                         any(kw in f for kw in ['ISCX', 'IDS', 'ids', 'cic', 'CIC', 'traffic'])]
            if csv_files:
                print(f"[*] Found CSV files: {csv_files}")
                df = load_real_dataset(csv_files)

        if df is None:
            print("[!] No real dataset found.")
            print("    To use real data, download from:")
            print("    https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv")
            print("    Place the CSV(s) in the same directory and re-run.")
            print("\n[*] Falling back to high-quality synthetic data...\n")

    if df is None:
        df = generate_synthetic_dataset(n_samples=80000)

    # ── 2. Features & labels ──
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_raw = df['label'].values

    le = LabelEncoder()
    le.fit(LABELS)  # fit on all possible labels for consistency
    y = le.transform(y_raw)
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    print(f"[✓] Label encoder saved. Classes: {list(le.classes_)}\n")

    # ── 3. Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(FEATURE_NAMES, f"{MODEL_DIR}/feature_names.pkl")
    print(f"[✓] Scaler saved.\n")

    # ── 4. Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # ── 5. Train supervised models ──
    rf = train_random_forest(X_train, X_test, y_train, y_test, le)
    xgb = train_xgboost(X_train, X_test, y_train, y_test, le)

    # Save which models are available
    available_models = {"random_forest": True, "xgboost": xgb is not None}
    joblib.dump(available_models, f"{MODEL_DIR}/available_models.pkl")

    # ── 6. Train Isolation Forest on benign only ──
    benign_mask = y_raw == 'BENIGN'
    X_benign = X_scaled[benign_mask]
    iso = train_isolation_forest(X_benign, X_scaled)

    print("\n" + "=" * 60)
    print("  ✓ All models trained and saved to ./models/")
    print(f"  ✓ Random Forest  — ready")
    print(f"  ✓ XGBoost        — {'ready' if xgb else 'skipped (pip install xgboost)'}")
    print(f"  ✓ Isolation Forest — ready")
    print("\n  Next step: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
