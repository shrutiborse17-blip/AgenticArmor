"""
AgenticArmor — Real ML Model Training
======================================
Trains two models:
  1. Random Forest  → supervised, classifies known attack types
  2. Isolation Forest → unsupervised, flags zero-day anomalies

Dataset priority:
  1. Real CIC-IDS-2018 CSVs in ./dataset/   ← BEST
  2. Kaggle auto-download (needs kaggle.json) ← AUTO
  3. Enhanced synthetic fallback             ← LAST RESORT

Run:  python train_models.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
import glob

MODEL_DIR   = "models"
DATASET_DIR = "dataset"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ── CIC-IDS-2018 column name mapping ────────────────────────────────────
# The real CSV has spaces — we strip & map to internal names.
CIC_COLUMN_MAP = {
    'Dst Port':'dst_port','Protocol':'protocol','Flow Duration':'flow_duration',
    'Tot Fwd Pkts':'tot_fwd_pkts','Tot Bwd Pkts':'tot_bwd_pkts',
    'TotLen Fwd Pkts':'totlen_fwd_pkts','TotLen Bwd Pkts':'totlen_bwd_pkts',
    'Fwd Pkt Len Max':'fwd_pkt_len_max','Fwd Pkt Len Min':'fwd_pkt_len_min',
    'Fwd Pkt Len Mean':'fwd_pkt_len_mean','Fwd Pkt Len Std':'fwd_pkt_len_std',
    'Bwd Pkt Len Max':'bwd_pkt_len_max','Bwd Pkt Len Min':'bwd_pkt_len_min',
    'Bwd Pkt Len Mean':'bwd_pkt_len_mean','Bwd Pkt Len Std':'bwd_pkt_len_std',
    'Flow Byts/s':'flow_byts_s','Flow Pkts/s':'flow_pkts_s',
    'Flow IAT Mean':'flow_iat_mean','Flow IAT Std':'flow_iat_std',
    'Flow IAT Max':'flow_iat_max','Flow IAT Min':'flow_iat_min',
    'Fwd IAT Tot':'fwd_iat_tot','Fwd IAT Mean':'fwd_iat_mean',
    'Fwd IAT Std':'fwd_iat_std','Fwd IAT Max':'fwd_iat_max','Fwd IAT Min':'fwd_iat_min',
    'Bwd IAT Tot':'bwd_iat_tot','Bwd IAT Mean':'bwd_iat_mean',
    'Bwd IAT Std':'bwd_iat_std','Bwd IAT Max':'bwd_iat_max','Bwd IAT Min':'bwd_iat_min',
    'Fwd PSH Flags':'fwd_psh_flags','Bwd PSH Flags':'bwd_psh_flags',
    'Fwd URG Flags':'fwd_urg_flags','Bwd URG Flags':'bwd_urg_flags',
    'Fwd Header Len':'fwd_header_len','Bwd Header Len':'bwd_header_len',
    'Fwd Pkts/s':'fwd_pkts_s','Bwd Pkts/s':'bwd_pkts_s',
    'Pkt Len Min':'pkt_len_min','Pkt Len Max':'pkt_len_max',
    'Pkt Len Mean':'pkt_len_mean','Pkt Len Std':'pkt_len_std','Pkt Len Var':'pkt_len_var',
    'FIN Flag Cnt':'fin_flag_cnt','SYN Flag Cnt':'syn_flag_cnt',
    'RST Flag Cnt':'rst_flag_cnt','PSH Flag Cnt':'psh_flag_cnt',
    'ACK Flag Cnt':'ack_flag_cnt','URG Flag Cnt':'urg_flag_cnt',
    'CWE Flag Count':'cwe_flag_count','ECE Flag Cnt':'ece_flag_cnt',
    'Down/Up Ratio':'down_up_ratio','Pkt Size Avg':'pkt_size_avg',
    'Fwd Seg Size Avg':'fwd_seg_size_avg','Bwd Seg Size Avg':'bwd_seg_size_avg',
    'Fwd Byts/b Avg':'fwd_byts_b_avg','Fwd Pkts/b Avg':'fwd_pkts_b_avg',
    'Fwd Blk Rate Avg':'fwd_blk_rate_avg','Bwd Byts/b Avg':'bwd_byts_b_avg',
    'Bwd Pkts/b Avg':'bwd_pkts_b_avg','Bwd Blk Rate Avg':'bwd_blk_rate_avg',
    'Subflow Fwd Pkts':'subflow_fwd_pkts','Subflow Fwd Byts':'subflow_fwd_byts',
    'Subflow Bwd Pkts':'subflow_bwd_pkts','Subflow Bwd Byts':'subflow_bwd_byts',
    'Init Fwd Win Byts':'init_fwd_win_byts','Init Bwd Win Byts':'init_bwd_win_byts',
    'Fwd Act Data Pkts':'fwd_act_data_pkts','Fwd Seg Size Min':'fwd_seg_size_min',
    'Active Mean':'active_mean','Active Std':'active_std',
    'Active Max':'active_max','Active Min':'active_min',
    'Idle Mean':'idle_mean','Idle Std':'idle_std',
    'Idle Max':'idle_max','Idle Min':'idle_min',
}
FEATURE_NAMES = list(CIC_COLUMN_MAP.values())

# Maps raw CIC-IDS-2018 labels → our internal labels
LABEL_MAP = {
    'Benign':'BENIGN','BENIGN':'BENIGN',
    'DDoS attacks-LOIC-HTTP':'DDoS','DDOS attack-LOIC-UDP':'DDoS',
    'DDOS attack-HOIC':'DDoS','DoS attacks-Hulk':'DDoS',
    'DoS attacks-SlowHTTPTest':'DDoS','DoS attacks-GoldenEye':'DDoS',
    'DoS attacks-Slowloris':'DDoS',
    'Bot':'C2_Heartbeat','Infilteration':'C2_Heartbeat',
    'SSH-Bruteforce':'BruteForce','FTP-BruteForce':'BruteForce',
    'Brute Force -Web':'BruteForce','Brute Force -XSS':'BruteForce',
    'SQL Injection':'SQLInjection',
}
LABELS = ['BENIGN', 'DDoS', 'PortScan', 'BruteForce', 'SQLInjection', 'C2_Heartbeat']


# ── LOAD REAL DATASET ────────────────────────────────────────────────────
def try_load_real_dataset():
    """
    Looks for CIC-IDS-2018 CSVs in ./dataset/
    Download from Kaggle: solarmainframe/ids-intrusion-csv
    Then place the CSV files into ./dataset/ and re-run this script.
    """
    csvs = glob.glob(os.path.join(DATASET_DIR, "*.csv")) + \
           glob.glob(os.path.join(DATASET_DIR, "**/*.csv"), recursive=True)
    if not csvs:
        return None

    print(f"[*] Found {len(csvs)} CSV file(s) in ./dataset/")
    dfs = []
    for path in csvs:
        try:
            df = pd.read_csv(path, nrows=150000, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"    [✓] {os.path.basename(path)}: {len(df):,} rows")
        except Exception as e:
            print(f"    [!] Failed {path}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns=CIC_COLUMN_MAP, inplace=True)

    # Find and map label column
    label_col = next((c for c in ['Label', 'label', ' Label'] if c in df.columns), None)
    if label_col is None:
        print("[!] No label column found.")
        return None

    df['label'] = df[label_col].astype(str).str.strip().map(LABEL_MAP)
    df = df[df['label'].notna()]

    # Ensure all feature cols exist (zero-fill missing)
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0

    df = df[FEATURE_NAMES + ['label']].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[✓] Real dataset: {len(df):,} rows\n{df['label'].value_counts()}\n")
    return df


# ── KAGGLE AUTO-DOWNLOAD ─────────────────────────────────────────────────
def try_kaggle_download():
    """
    Needs kaggle.json at: C:\\Users\\<you>\\.kaggle\\kaggle.json
    Get it from: https://www.kaggle.com/settings → API → Create New Token
    """
    try:
        import kaggle
        print("[*] Kaggle API found — attempting download...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'solarmainframe/ids-intrusion-csv',
            path=DATASET_DIR, unzip=True
        )
        print(f"[✓] Downloaded to ./{DATASET_DIR}/")
        return True
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("    To enable: place kaggle.json in C:\\Users\\<you>\\.kaggle\\")
        return False


# ── SYNTHETIC FALLBACK ───────────────────────────────────────────────────
def generate_synthetic_dataset(n_samples=80000):
    print(f"[*] Generating {n_samples:,} enhanced synthetic samples...")
    np.random.seed(42)
    rows = []
    spc  = n_samples // len(LABELS)

    for label in LABELS:
        for _ in range(spc):
            f = np.zeros(len(FEATURE_NAMES))
            if label == 'BENIGN':
                f[0]=np.random.choice([80,443,53,123,8080,22,25,587])
                f[1]=np.random.choice([6,17]); f[2]=np.random.randint(1000,200000)
                f[3]=np.random.randint(2,80);  f[4]=np.random.randint(2,80)
                f[5]=np.random.randint(200,10000); f[16]=np.random.uniform(10,800)
                f[17]=np.random.uniform(1000,50000); f[48]=np.random.randint(0,3)
                f[52]=np.random.randint(2,20); f[65]=np.random.randint(8192,65535)
            elif label == 'DDoS':
                f[0]=np.random.choice([80,443,53]); f[1]=np.random.choice([17,6])
                f[2]=np.random.randint(50,3000); f[3]=np.random.randint(5000,30000)
                f[4]=np.random.randint(0,3); f[5]=np.random.randint(200000,2000000)
                f[16]=np.random.uniform(5000,30000); f[17]=np.random.uniform(10,500)
                f[44]=0; f[48]=np.random.randint(2000,10000); f[7]=np.random.randint(40,64)
            elif label == 'PortScan':
                f[0]=np.random.randint(1,65535); f[1]=6
                f[2]=np.random.randint(5,300); f[3]=np.random.randint(1,3)
                f[5]=np.random.randint(40,80); f[48]=1; f[52]=0
                f[45]=np.random.randint(0,2); f[65]=0; f[66]=0
            elif label == 'BruteForce':
                f[0]=np.random.choice([22,21,3389,23,5900]); f[1]=6
                f[2]=np.random.randint(300,5000); f[3]=np.random.randint(8,60)
                f[4]=np.random.randint(8,60); f[5]=np.random.randint(800,6000)
                f[16]=np.random.uniform(15,150); f[48]=np.random.randint(1,4)
                f[52]=np.random.randint(8,50); f[17]=np.random.uniform(500,2000)
                f[18]=np.random.uniform(10,100)
            elif label == 'SQLInjection':
                f[0]=np.random.choice([3306,5432,1433,80,443,8080]); f[1]=6
                f[2]=np.random.randint(8000,80000); f[3]=np.random.randint(4,25)
                f[5]=np.random.randint(8000,80000); f[7]=np.random.randint(1000,8000)
                f[16]=np.random.uniform(1,15); f[51]=np.random.randint(5,25)
            elif label == 'C2_Heartbeat':
                f[0]=np.random.choice([443,8443,4444,8080,8888]); f[1]=6
                f[2]=np.random.randint(27000,33000); f[3]=np.random.randint(2,5)
                f[4]=np.random.randint(2,5); f[5]=np.random.randint(108,132)
                f[7]=np.random.randint(112,126); f[16]=np.random.uniform(0.05,0.35)
                f[17]=np.random.uniform(27000,33000); f[18]=np.random.uniform(10,200)

            f += np.random.normal(0, 0.3, len(FEATURE_NAMES))
            rows.append((*np.clip(f, 0, None), label))

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ['label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[✓] Synthetic dataset: {df.shape}\n{df['label'].value_counts()}\n")
    return df


# ── TRAIN ────────────────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("[*] Training Random Forest (200 trees, balanced)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_split=4,
        min_samples_leaf=2, n_jobs=-1, random_state=42,
        class_weight='balanced', oob_score=True,
    )
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"[✓] Accuracy: {acc*100:.2f}%  |  OOB: {rf.oob_score_*100:.2f}%")
    print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))
    joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
    return rf


def train_isolation_forest(X_benign):
    print(f"[*] Training Isolation Forest on {len(X_benign):,} benign samples...")
    iso = IsolationForest(
        n_estimators=300, contamination=0.04,
        max_samples=min(256, len(X_benign)),
        max_features=0.8, random_state=42, n_jobs=-1,
    )
    iso.fit(X_benign)
    print(f"[✓] Isolation Forest ready")
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest.pkl")
    return iso


# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  AgenticArmor — ML Training Pipeline")
    print("=" * 58 + "\n")

    # Priority: real CSV → Kaggle download → synthetic
    df = try_load_real_dataset()
    if df is None:
        if try_kaggle_download():
            df = try_load_real_dataset()
    if df is None:
        print("\n[!] Using synthetic data. For real data:")
        print("    1. Download from: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv")
        print("    2. Place CSV files into ./dataset/")
        print("    3. Re-run this script\n")
        df = generate_synthetic_dataset(80000)

    X     = df[FEATURE_NAMES].values.astype(np.float32)
    y_raw = df['label'].values
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(le,            f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(scaler,        f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(FEATURE_NAMES, f"{MODEL_DIR}/feature_names.pkl")
    print(f"[✓] Classes: {list(le.classes_)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    train_random_forest(X_train, X_test, y_train, y_test, le)
    train_isolation_forest(X_scaled[y_raw == 'BENIGN'])

    print("=" * 58)
    print("  ✓ All models saved to ./models/")
    print("  Next step: python main.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
