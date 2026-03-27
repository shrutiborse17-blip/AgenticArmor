# AgenticArmor — Production Setup Guide
## Innovate Maharashtra 2026

---

## WHAT'S REAL NOW vs THE ORIGINAL

| Component | Original (Demo) | Now (Real) |
|---|---|---|
| Training data | Synthetic only | Real CIC-IDS-2018 dataset supported |
| ML models | Random Forest only | Random Forest + **XGBoost ensemble** |
| Packet capture | Fake numbers | **Real NIC capture via Scapy** |
| Alert storage | In-memory (lost on restart) | **SQLite persistent database** |
| Auto-response | Printed text only | **Auto IP block + rate limiting** |
| Gemini | 1.5 Flash | **Gemini 2.0 Flash** |
| Threat logic | Basic | **Ensemble voting + auto-block** |

---

## PROJECT STRUCTURE

```
agenticarmor/
├── train_models.py       ← Step 1: Train real ML models
├── main.py               ← Step 2: Start FastAPI backend (production-grade)
├── capture.py            ← Step 3a: REAL packet capture (use this for live demo)
├── simulate_traffic.py   ← Step 3b: Test mode only (synthetic data)
├── dashboard_live.html   ← Step 4: Open in browser
├── requirements.txt      ← Python dependencies
└── models/               ← Auto-created by train_models.py
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── isolation_forest.pkl
    ├── label_encoder.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## FULL SETUP (15–20 MINUTES)

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

**Also install Npcap** (required for live packet capture on Windows):
- Download from: https://npcap.com/#download
- During install: ✅ Check "Install Npcap in WinPcap API-compatible Mode"

---

### Step 2 — Get the Real Dataset (Recommended)

Download CIC-IDS-2018 from Kaggle:
```
https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
```

Place the CSV file(s) in the same folder as `train_models.py`.
The script will auto-detect any CSV with "ISCX", "IDS", or "cic" in the name.

---

### Step 3 — Train the Models

**With real dataset (recommended):**
```bash
python train_models.py
```

**Without dataset (synthetic fallback):**
```bash
python train_models.py --synthetic
```

**Point to specific CSV:**
```bash
python train_models.py --csv Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Expected output:
```
[✓] Random Forest Accuracy: 97–99%
[✓] XGBoost Accuracy: 98–99%
[✓] All models saved to ./models/
```

---

### Step 4 — Start the Backend

**Without Gemini:**
```bash
python main.py
```

**With Gemini 2.0 Flash (recommended):**
```cmd
set GEMINI_API_KEY=your_key_here
python main.py
```

Get a free Gemini API key: https://aistudio.google.com/apikey

Server starts at: http://localhost:8000
API docs at:      http://localhost:8000/docs

---

### Step 5 — Open the Dashboard

Open `dashboard_live.html` in Chrome.
Click **CONNECT** → then **START LIVE FEED**.

---

### Step 6 — Start Real Packet Capture

Open a **new terminal as Administrator** and run:

```cmd
# See your interfaces first
python capture.py --list-ifaces

# Capture on Wi-Fi
python capture.py --iface "Wi-Fi"

# Capture on Ethernet
python capture.py --iface "Ethernet"

# Capture only TCP traffic
python capture.py --iface "Wi-Fi" --bpf "tcp"
```

> ⚠️ **Must run as Administrator** for raw packet capture.
> Right-click terminal → "Run as administrator"

---

### Alternative: Test Mode (no Npcap needed)

If you just want to test the pipeline without real capture:

```bash
python simulate_traffic.py
python simulate_traffic.py --attack DDoS
python simulate_traffic.py --rate 0.5 --count 100
```

---

## HOW IT WORKS (REAL SYSTEM)

```
Your NIC (Wi-Fi / Ethernet)
      │
      │  Raw packets (Scapy)
      ▼
capture.py
  ├── Builds bidirectional flows (5s timeout)
  ├── Extracts 78 CIC-IDS-2018 features per flow
  └── POST /analyze

main.py (FastAPI)
  ├── Check block list → reject if blocked
  ├── StandardScaler.transform()
  ├── RandomForest.predict_proba()    ──→ Label + Confidence
  ├── XGBoost.predict_proba()         ──→ Label + Confidence
  ├── Ensemble vote                   ──→ Final prediction
  ├── IsolationForest.score_samples() ──→ Anomaly score
  ├── Auto-block CRITICAL threats
  ├── Rate-limit repeat offenders
  ├── Gemini 2.0 Flash narrative
  ├── Remediation playbook
  └── Save to SQLite (agenticarmor.db)
      │
      │  JSON response
      ▼
dashboard_live.html
  └── Live feed, detail panel, remediation log, stats
```

---

## API ENDPOINTS

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Model status, Gemini status |
| /analyze | POST | Analyze a traffic flow (78 features) |
| /stats | GET | Session statistics |
| /alerts | GET | Persistent alert history from DB |
| /blocklist | GET | Currently blocked IPs |
| /blocklist/{ip} | DELETE | Manually unblock an IP |
| /features | GET | Feature names list |
| /docs | GET | Interactive API docs |

---

## AUTO-BLOCKING RULES

The backend automatically blocks source IPs:

| Condition | Action | Duration |
|---|---|---|
| CRITICAL threat (DDoS/C2, >85% confidence) | Auto-block | 2 hours |
| HIGH threat + rate limit exceeded | Auto-block | 30 min |

Unblock manually:
```bash
curl -X DELETE http://localhost:8000/blocklist/192.168.1.5
```

---

## USING THE REAL CIC-IDS-2018 DATASET

The dataset contains real labelled network traffic captures from:
- CICIDS 2018 (Canadian Institute for Cybersecurity)
- ~16 million flows across multiple attack types

Column mapping is handled automatically. If a column name doesn't match,
the script fills it with 0 and continues.

Label mapping:
```
DoS Hulk / GoldenEye / slowloris → DDoS
FTP-Patator / SSH-Patator         → BruteForce
Bot / Infiltration / Heartbleed   → C2_Heartbeat
Web Attack Sql Injection          → SQLInjection
Web Attack XSS                    → SQLInjection
```

---

## HACKATHON DEMO SCRIPT

1. Open dashboard fullscreen in Chrome
2. Start backend: `python main.py` (with GEMINI_API_KEY set)
3. Open **Admin terminal** → `python capture.py --iface "Wi-Fi"`
4. Dashboard shows CONNECT → **LIVE ML BACKEND CONNECTED**
5. Real traffic starts flowing — mostly BENIGN (green)
6. In second terminal: `python simulate_traffic.py --attack DDoS`
   → Watch red DDoS alerts hit the dashboard
7. Click a red row → show judges:
   - **Real flow features** extracted from your NIC
   - **RF + XGBoost ensemble** verdict with confidence
   - **ISO anomaly score** (zero-day detection)
   - **Gemini 2.0 Flash** narrative (AI-generated)
   - **Auto-block** rule deployed
   - **SQLite alert** saved (show agenticarmor.db in DB Browser)
8. Open http://localhost:8000/alerts → show persistent alert log

---

## TROUBLESHOOTING

**"Scapy not installed"**
```bash
pip install scapy
```

**"Permission denied" / capture not working**
→ Run terminal as Administrator

**"No packets captured"**
```bash
python capture.py --list-ifaces
# Try different interface names
```

**"Models not found"**
```bash
python train_models.py --synthetic
```

**"Gemini error"**
→ Check GEMINI_API_KEY env var is set correctly
→ Get key from https://aistudio.google.com/apikey

---

## TEAM ROLES

| Person | Files to own |
|---|---|
| ML Engineer | train_models.py |
| Backend Dev | main.py, capture.py |
| AI Engineer | Gemini prompt in main.py |
| Frontend Dev | dashboard_live.html |

---

Good luck! — AgenticArmor Team, Innovate Maharashtra 2026
