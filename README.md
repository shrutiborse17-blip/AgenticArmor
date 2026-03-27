# AgenticArmor — Real System Setup Guide

## What's New vs Demo Version

| Feature | Demo | Real |
|---|---|---|
| Training data | Synthetic | Real CIC-IDS-2018 or Kaggle auto-download |
| Packet capture | Fake numbers | Live NIC capture via Scapy |
| Alert storage | RAM only (lost on restart) | SQLite persistent (`alerts.db`) |
| API keys | env var only | `.env` file |
| Alert history | None | `/alerts` endpoint with filters |
| Repeat offender detection | No | Yes — escalates to CRITICAL |
| Gemini narratives | Optional | Configured via `.env` |

---

## PROJECT STRUCTURE

```
agenticarmor/
├── train_models.py       ← Step 1: Train ML models
├── main.py               ← Step 2: FastAPI backend
├── capture.py            ← Step 3a: REAL live capture (NEW)
├── simulate_traffic.py   ← Step 3b: Demo/synthetic mode
├── dashboard_live.html   ← Step 4: Browser dashboard
├── requirements.txt      ← Dependencies
├── .env                  ← API keys (create this)
├── dataset/              ← Place CIC-IDS-2018 CSVs here
├── models/               ← Auto-created by train_models.py
└── alerts.db             ← Auto-created by main.py
```

---

## SETUP

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

For real packet capture, also install **Npcap**:
- Download from: https://npcap.com/#download
- During install, check **"WinPcap API-compatible Mode"**

---

### Step 2 — Configure API Keys

Create a `.env` file in the project folder:

```
GEMINI_API_KEY=your_key_here
```

Get a free Gemini key at: https://aistudio.google.com/app/apikey

---

### Step 3 — Get Real Training Data (Recommended)

**Option A — Manual download:**
1. Go to: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
2. Download any CSV files (e.g. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
3. Place them in `./dataset/`
4. Run `python train_models.py` — it will auto-detect them

**Option B — Kaggle auto-download:**
1. Go to: https://www.kaggle.com/settings → API → Create New Token
2. Place `kaggle.json` at `C:\Users\<you>\.kaggle\kaggle.json`
3. Run `python train_models.py` — it will download automatically

**Option C — Skip (uses enhanced synthetic data):**
Just run `python train_models.py` without any CSV files.

---

### Step 4 — Train Models

```bash
python train_models.py
```

Expected output:
```
[✓] Random Forest Accuracy: 97-99%  (real data)  or  95-98% (synthetic)
[✓] All models saved to ./models/
```

---

### Step 5 — Start Backend

```bash
python main.py
```

Server: http://localhost:8000
API docs: http://localhost:8000/docs

---

### Step 6 — Start Detection

**Real mode (live NIC capture):**
```bash
# Run as Administrator!
python capture.py --list              # see interfaces
python capture.py --iface "Wi-Fi"    # capture Wi-Fi traffic
python capture.py                     # auto-detect interface
```

**Demo mode (synthetic packets):**
```bash
python simulate_traffic.py
python simulate_traffic.py --attack DDoS
python simulate_traffic.py --rate 0.3
```

---

### Step 7 — Open Dashboard

Open `dashboard_live.html` in Chrome.
Click **CONNECT** → **START LIVE FEED**.

---

## API ENDPOINTS

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Status + model/Gemini check |
| /analyze | POST | Analyze a traffic flow |
| /alerts | GET | Persistent alert history |
| /alerts?level=CRITICAL | GET | Filter by threat level |
| /alerts?type=DDoS | GET | Filter by attack type |
| /top_threats | GET | Most frequent source IPs |
| /stats | GET | Session + DB statistics |
| /features | GET | Feature names list |
| /docs | GET | Interactive API docs |

---

## ARCHITECTURE

```
[Real NIC]
    │
    ▼
capture.py  (Scapy — flow tracking, 78 feature extraction)
    │
    │  POST /analyze
    ▼
main.py (FastAPI)
    ├── StandardScaler.transform()
    ├── RandomForest.predict_proba()    → Label + Confidence
    ├── IsolationForest.score_samples() → Anomaly score
    ├── Repeat-offender check           → Escalation
    ├── Gemini 1.5 Flash narrative      → AI explanation
    ├── Remediation playbook            → Actions
    └── SQLite log                      → Persistent storage
    │
    │  JSON response
    ▼
dashboard_live.html  (Live feed, detail panel, alert history)
```

---

## HACKATHON DEMO TIPS

1. Open dashboard fullscreen in Chrome
2. Run `python capture.py` (as Admin) in a terminal
3. Open another terminal: `python simulate_traffic.py --attack DDoS`
4. Watch dashboard react with real ML verdicts
5. Show judges the `/alerts` endpoint in Postman or browser
6. Point out that alerts.db persists across restarts (unlike the demo)

Good luck — AgenticArmor Team, Innovate Maharashtra 2026!
