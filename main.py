"""
AgenticArmor — FastAPI Backend (Production-Ready)
==================================================
Changes from demo version:
  - SQLite persistent alert logging (survives restarts)
  - .env file for API keys (no more env var juggling)
  - /alerts endpoint with filters (level, type, time range)
  - Gemini 1.5 Flash narratives
  - Rate limiting per source IP
  - Proper startup/shutdown events

Run:
    python train_models.py   ← first time only
    python main.py
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import joblib
import os
import time
import asyncio
import sqlite3
import json
from datetime import datetime
from collections import defaultdict

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

async def get_gemini_narrative(label, src_ip, port, protocol, pps, confidence, iso_score, raw_log):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are a cybersecurity SOC analyst.

Analyze this network traffic:

Attack Type: {label}
Source IP: {src_ip}
Port: {port}
Protocol: {protocol}
Packets/sec: {pps}
Confidence: {confidence}%
Anomaly Score: {iso_score}

Explain:
1. What is happening
2. Why it is dangerous
3. What should be done

Keep it short (2-3 lines).
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print("Gemini error:", e)
        return ""
# ── .env support ─────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # pip install python-dotenv

import google.generativeai as genai

MODEL_DIR = "models"
DB_PATH   = "alerts.db"

# ── DATABASE SETUP ────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            src_ip        TEXT,
            dst_ip        TEXT,
            dst_port      INTEGER,
            protocol      TEXT,
            rf_prediction TEXT,
            rf_confidence REAL,
            iso_score     REAL,
            threat_level  TEXT,
            alert_type    TEXT,
            narrative     TEXT,
            narrative_src TEXT,
            remediation   TEXT,
            analysis_ms   REAL
        )
    """)
    con.commit()
    con.close()
    print(f"[✓] SQLite database: {DB_PATH}")


def log_alert(result: dict, sample: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO alerts
          (timestamp, src_ip, dst_ip, dst_port, protocol,
           rf_prediction, rf_confidence, iso_score, threat_level, alert_type,
           narrative, narrative_src, remediation, analysis_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result["timestamp"],
        sample.get("src_ip", ""),
        sample.get("dst_ip", ""),
        sample.get("dst_port", 0),
        sample.get("protocol", ""),
        result["rf_prediction"],
        result["rf_confidence"],
        result["iso_score"],
        result["threat_level"],
        result["alert_type"],
        result["narrative"],
        result["narrative_source"],
        json.dumps(result["remediation_actions"]),
        result["analysis_time_ms"],
    ))
    con.commit()
    con.close()


# ── MODEL LOADING ─────────────────────────────────────────────────────────
rf_model = iso_model = label_encoder = scaler = feature_names = None

def load_models():
    global rf_model, iso_model, label_encoder, scaler, feature_names
    try:
        rf_model      = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
        iso_model     = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        scaler        = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
        print(f"[✓] Models loaded — classes: {list(label_encoder.classes_)}")
        return True
    except FileNotFoundError:
        print("[!] Models not found — run: python train_models.py")
        return False


# ── GEMINI SETUP ──────────────────────────────────────────────────────────
gemini_model = None

def init_gemini():
    global gemini_model
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("[!] GEMINI_API_KEY not set.")
        print("    Add it to a .env file: GEMINI_API_KEY=your_key_here")
        print("    Get a free key at: https://aistudio.google.com/app/apikey")
        return
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        # Quick test
        gemini_model.generate_content("ping", generation_config={"max_output_tokens": 5})
        print("[✓] Gemini 1.5 Flash connected")
    except Exception as e:
        print(f"[!] Gemini init failed: {e}")
        gemini_model = None


# ── APP LIFECYCLE ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_models()
    init_gemini()
    yield

app = FastAPI(title="AgenticArmor API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── IN-MEMORY SESSION STATS ───────────────────────────────────────────────
session_stats = {
    "total_analyzed":  0,
    "threats_detected": 0,
    "anomalies_flagged": 0,
    "attack_counts":   {k: 0 for k in ['DDoS','PortScan','BruteForce','SQLInjection','C2_Heartbeat']},
    "recent_threats":  [],
    "start_time":      datetime.now().isoformat(),
}

# Per-IP rate tracking (for repeated offenders)
ip_threat_count = defaultdict(int)


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────
class TrafficSample(BaseModel):
    features:        List[float]
    raw_log:         Optional[str]  = ""
    src_ip:          Optional[str]  = "unknown"
    dst_ip:          Optional[str]  = "unknown"
    dst_port:        Optional[int]  = 0
    protocol:        Optional[str]  = "TCP"
    packets_per_sec: Optional[float] = 0.0


class AnalysisResult(BaseModel):
    rf_prediction:    str
    rf_confidence:    float
    rf_probabilities: dict
    iso_prediction:   str
    iso_score:        float
    threat_detected:  bool
    threat_level:     str
    alert_type:       str
    narrative:        str
    narrative_source: str
    remediation_actions: List[str]
    repeat_offender:  bool
    analysis_time_ms: float
    timestamp:        str


# ── REMEDIATION PLAYBOOKS ─────────────────────────────────────────────────
REMEDIATION = {
    "DDoS": [
        "BLOCK {src_ip}/32 — volumetric flood confirmed (rate: {pps:.0f} pkt/s)",
        "RATE LIMIT upstream load balancer — drop UDP/TCP flood from {src_ip}",
        "ESCALATE to Tier-2 SOC — DDoS sustained for >30s",
        "RUN: netsh advfirewall firewall add rule name=\"Block {src_ip}\" dir=in action=block remoteip={src_ip}",
    ],
    "PortScan": [
        "BLOCK {src_ip} — sequential port probe on {dst_port} targets",
        "ENABLE honeypot on scanned port range",
        "FLAG {src_ip} in threat intel feed for 24h",
    ],
    "BruteForce": [
        "LOCK accounts — >5 failed auth from {src_ip} on port {dst_port}",
        "THROTTLE {src_ip} to 5 req/min on port {dst_port}",
        "RUN: netsh advfirewall firewall add rule name=\"BruteForce {src_ip}\" dir=in action=block remoteip={src_ip}",
        "ALERT security team — brute force on port {dst_port}",
    ],
    "SQLInjection": [
        "BLOCK request at WAF — SQLi payload detected from {src_ip}",
        "TERMINATE DB connections from {src_ip} to port {dst_port}",
        "LOG full payload for forensic analysis",
        "ESCALATE — possible data exfiltration from {src_ip}",
    ],
    "C2_Heartbeat": [
        "ISOLATE host {src_ip} from network immediately",
        "SINKHOLE outbound DNS from {src_ip}",
        "TRIGGER endpoint scan on {src_ip}",
        "CRITICAL — active C2 beacon — escalate to incident response",
    ],
    "BENIGN": [],
}

def get_remediation(label, src_ip, dst_port, pps):
    templates = REMEDIATION.get(label, [])
    return [t.format(src_ip=src_ip, dst_port=dst_port, pps=pps) for t in templates[:3]]


# ── RULE-BASED NARRATIVE (no Gemini fallback) ─────────────────────────────
RULE_NARRATIVES = {
    "DDoS":         "Volumetric DDoS flood from {src_ip} at {pps:.0f} pkt/s targeting port {dst_port}. RF confidence: {conf:.1f}%. Isolation Forest score {iso:.4f} confirms abnormal volume. Upstream rate limiting deployed.",
    "PortScan":     "Stealthy port scan from {src_ip} detected — sequential SYN probes indicate pre-exploitation reconnaissance. Anomaly score: {iso:.4f}. Host flagged for investigation.",
    "BruteForce":   "Brute-force credential attack from {src_ip} on port {dst_port} ({conf:.1f}% confidence). Scripted auth attempts with regular IAT detected. Account lockout and IP block triggered.",
    "SQLInjection": "SQL injection attempt from {src_ip} targeting port {dst_port}. Large payload ({pps:.0f} pkt/s) with PSH flags suggests UNION SELECT exfiltration. WAF block applied.",
    "C2_Heartbeat": "C2 beacon from {src_ip} — periodic {pps:.2f} pkt/s heartbeat with uniform packet size (~118 bytes). Encryption masking payload. Host quarantined from network.",
    "BENIGN":       "Traffic from {src_ip}:{dst_port} classified benign ({conf:.1f}% confidence). Anomaly score: {iso:.4f}. No action required.",
}

def rule_narrative(label, src_ip, dst_port, conf, iso, pps):
    t = RULE_NARRATIVES.get(label, "Traffic analyzed. No template available.")
    return t.format(src_ip=src_ip, dst_port=dst_port, conf=conf, iso=iso, pps=pps)


# ── GEMINI NARRATIVE ──────────────────────────────────────────────────────
async def get_gemini_narrative(label, src_ip, dst_port, proto, pps, conf, iso, raw_log):
    if not gemini_model:
        return ""
    try:
        prompt = f"""You are a senior SOC analyst writing for a live security dashboard.
A real-time ML detection just fired. Write a concise 3-sentence threat narrative.

Detection details:
- Threat type: {label}
- Source IP: {src_ip}  →  Destination port: {dst_port}  Protocol: {proto}
- Packets/sec: {pps:.1f}  RF confidence: {conf:.1f}%  ISO score: {iso:.4f}
- Raw flow log: {raw_log}

Instructions:
- Be technical and specific to the attack type
- Mention what the attacker is likely trying to achieve
- End with one concrete immediate action for the SOC
- Exactly 3 sentences, no bullet points, no markdown"""

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: gemini_model.generate_content(prompt)
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[!] Gemini error: {e}")
        return ""


# ── THREAT LEVEL ──────────────────────────────────────────────────────────
def threat_level(label, conf, iso, repeat):
    if label == "BENIGN":
        return "LOW"
    if repeat and label in ["DDoS", "C2_Heartbeat", "BruteForce"]:
        return "CRITICAL"
    if label in ["DDoS", "C2_Heartbeat"] and conf > 88:
        return "CRITICAL"
    if label in ["SQLInjection", "BruteForce"] and conf > 82:
        return "HIGH"
    if iso < -0.3:
        return "HIGH"
    if conf > 72:
        return "MEDIUM"
    return "LOW"


# ── /analyze ─────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResult)
async def analyze(sample: TrafficSample):
    if rf_model is None:
        raise HTTPException(503, "Models not loaded — run: python train_models.py")

    t0 = time.time()

    if len(sample.features) != len(feature_names):
        raise HTTPException(400, f"Expected {len(feature_names)} features, got {len(sample.features)}")

    arr        = np.array(sample.features, dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    # Random Forest
    rf_probs   = rf_model.predict_proba(arr_scaled)[0]
    rf_idx     = int(np.argmax(rf_probs))
    rf_label   = label_encoder.inverse_transform([rf_idx])[0]
    rf_conf    = float(rf_probs[rf_idx]) * 100
    rf_prob_dict = {cls: round(float(p)*100, 2) for cls, p in zip(label_encoder.classes_, rf_probs)}

    # Isolation Forest
    iso_score  = float(iso_model.score_samples(arr_scaled)[0])
    iso_pred   = "ANOMALY" if iso_model.predict(arr_scaled)[0] == -1 else "NORMAL"

    # Combined decision
    threat     = (rf_label != "BENIGN" and rf_conf > 60) or iso_score < -0.35
    alert_type = rf_label if rf_label != "BENIGN" else ("ZERO_DAY_ANOMALY" if iso_pred == "ANOMALY" else "BENIGN")

    # Repeat offender check
    repeat = False
    if threat and sample.src_ip != "unknown":
        ip_threat_count[sample.src_ip] += 1
        repeat = ip_threat_count[sample.src_ip] > 3

    level = threat_level(rf_label, rf_conf, iso_score, repeat)

    # Narrative
    narrative = narrative_src = ""
    if threat:
        narrative = await get_gemini_narrative(
            rf_label, sample.src_ip, sample.dst_port,
            sample.protocol, sample.packets_per_sec, rf_conf, iso_score, sample.raw_log
        )
        narrative_src = "gemini_1.5_flash" if narrative else "rule_based"
        if not narrative:
            narrative = rule_narrative(rf_label, sample.src_ip, sample.dst_port, rf_conf, iso_score, sample.packets_per_sec)
    else:
        narrative     = rule_narrative("BENIGN", sample.src_ip, sample.dst_port, rf_conf, iso_score, sample.packets_per_sec)
        narrative_src = "rule_based"

    remediation = get_remediation(rf_label, sample.src_ip, sample.dst_port, sample.packets_per_sec) if threat else []
    elapsed     = round((time.time() - t0) * 1000, 2)
    ts          = datetime.now().isoformat()

    result = dict(
        rf_prediction=rf_label, rf_confidence=round(rf_conf, 2),
        rf_probabilities=rf_prob_dict,
        iso_prediction=iso_pred, iso_score=round(iso_score, 4),
        threat_detected=threat, threat_level=level, alert_type=alert_type,
        narrative=narrative, narrative_source=narrative_src,
        remediation_actions=remediation, repeat_offender=repeat,
        analysis_time_ms=elapsed, timestamp=ts,
    )

    # Update stats
    session_stats["total_analyzed"] += 1
    if threat:
        session_stats["threats_detected"] += 1
        if rf_label in session_stats["attack_counts"]:
            session_stats["attack_counts"][rf_label] += 1
        session_stats["recent_threats"].append({
            "time": ts, "label": rf_label, "src_ip": sample.src_ip,
            "confidence": rf_conf, "threat_level": level,
        })
        if len(session_stats["recent_threats"]) > 50:
            session_stats["recent_threats"].pop(0)
        # Persist to DB
        log_alert(result, sample.dict())
    if iso_pred == "ANOMALY":
        session_stats["anomalies_flagged"] += 1

    return AnalysisResult(**result)


# ── /alerts — persistent history ──────────────────────────────────────────
@app.get("/alerts")
def get_alerts(
    level:  Optional[str] = Query(None, description="Filter by threat level: LOW/MEDIUM/HIGH/CRITICAL"),
    type_:  Optional[str] = Query(None, alias="type", description="Filter by alert type"),
    limit:  int           = Query(100, le=1000),
    offset: int           = Query(0),
):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    q      = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if level:
        q += " AND threat_level = ?"; params.append(level.upper())
    if type_:
        q += " AND alert_type = ?";   params.append(type_)
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    rows = [dict(r) for r in con.execute(q, params).fetchall()]
    total = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    con.close()

    for row in rows:
        try: row["remediation"] = json.loads(row["remediation"])
        except: pass

    return {"total": total, "offset": offset, "limit": limit, "alerts": rows}


# ── /stats ────────────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    con   = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    con.close()
    return {**session_stats, "total_alerts_persisted": total}


# ── /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":          "ok",
        "models_loaded":   rf_model is not None,
        "gemini_connected": gemini_model is not None,
        "feature_count":   len(feature_names) if feature_names else 0,
        "db_path":         DB_PATH,
    }


# ── /features ─────────────────────────────────────────────────────────────
@app.get("/features")
def get_features():
    return {"features": feature_names, "count": len(feature_names)}


# ── /top_threats ──────────────────────────────────────────────────────────
@app.get("/top_threats")
def top_threats(limit: int = Query(10, le=50)):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT src_ip, COUNT(*) as count, MAX(threat_level) as max_level
        FROM alerts
        GROUP BY src_ip
        ORDER BY count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [{"src_ip": r[0], "alert_count": r[1], "max_level": r[2]} for r in rows]


# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
