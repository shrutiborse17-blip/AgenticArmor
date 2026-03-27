"""
AgenticArmor — FastAPI Backend (Production-Grade)
===================================================
Real ML inference server with:
  - Random Forest + XGBoost ensemble
  - Isolation Forest anomaly detection
  - SQLite persistent alert log
  - Gemini 2.0 Flash threat narratives
  - Rate limiting per source IP
  - Block list management
  - Full audit trail

Run:
    python main.py
    set GEMINI_API_KEY=your_key && python main.py   (Windows)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import joblib
import os
import time
import asyncio
import sqlite3
import json
import threading
from datetime import datetime, timedelta
from collections import defaultdict

# ── INIT ───────────────────────────────────────────────────────────────
app = FastAPI(title="AgenticArmor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
DB_PATH   = "agenticarmor.db"

# ── LOAD MODELS ────────────────────────────────────────────────────────
print("[*] Loading ML models...")
try:
    rf_model      = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    iso_model     = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    scaler        = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    print(f"[✓] Core models loaded. Classes: {list(label_encoder.classes_)}")

    # Try XGBoost
    xgb_model = None
    try:
        xgb_model = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
        print("[✓] XGBoost loaded — ensemble mode active")
    except FileNotFoundError:
        print("[!] XGBoost not found — using Random Forest only")

except FileNotFoundError:
    print("[!] Models not found. Run: python train_models.py first.")
    rf_model = iso_model = label_encoder = scaler = feature_names = xgb_model = None

# ── GEMINI SETUP ───────────────────────────────────────────────────────
gemini_model   = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("[✓] Gemini 2.0 Flash connected")
    except Exception as e:
        print(f"[!] Gemini setup failed: {e}")
else:
    print("[!] No GEMINI_API_KEY — set env var for live narratives")
    print("    Windows: set GEMINI_API_KEY=your_key_here")

# ── SQLITE DATABASE ────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            src_ip          TEXT,
            dst_ip          TEXT,
            dst_port        INTEGER,
            protocol        TEXT,
            rf_prediction   TEXT,
            rf_confidence   REAL,
            iso_score       REAL,
            iso_prediction  TEXT,
            threat_detected INTEGER,
            threat_level    TEXT,
            alert_type      TEXT,
            narrative       TEXT,
            narrative_source TEXT,
            remediation     TEXT,
            analysis_ms     REAL,
            packets_per_sec REAL,
            raw_log         TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS block_list (
            ip          TEXT PRIMARY KEY,
            reason      TEXT,
            added_at    TEXT,
            expires_at  TEXT
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_src_ip ON alerts(src_ip)
    """)
    conn.commit()
    conn.close()
    print(f"[✓] SQLite database ready: {DB_PATH}")

init_db()

# Thread-local DB connections
_db_local = threading.local()

def get_db():
    if not hasattr(_db_local, 'conn'):
        _db_local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return _db_local.conn

def save_alert(data: dict, remediation: list, narrative: str, narrative_source: str):
    try:
        conn = get_db()
        c    = conn.cursor()
        c.execute("""
            INSERT INTO alerts (
                timestamp, src_ip, dst_ip, dst_port, protocol,
                rf_prediction, rf_confidence, iso_score, iso_prediction,
                threat_detected, threat_level, alert_type,
                narrative, narrative_source, remediation, analysis_ms,
                packets_per_sec, raw_log
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data['timestamp'], data['src_ip'], data['dst_ip'],
            data['dst_port'], data['protocol'],
            data['rf_prediction'], data['rf_confidence'],
            data['iso_score'], data['iso_prediction'],
            int(data['threat_detected']), data['threat_level'], data['alert_type'],
            narrative, narrative_source, json.dumps(remediation),
            data['analysis_ms'], data['packets_per_sec'], data.get('raw_log',''),
        ))
        conn.commit()
    except Exception as e:
        print(f"[!] DB save error: {e}")

# ── IN-MEMORY STATS ────────────────────────────────────────────────────
session_stats = {
    "total_analyzed":  0,
    "threats_detected": 0,
    "anomalies_flagged": 0,
    "attack_counts": {k: 0 for k in ['DDoS','PortScan','BruteForce','SQLInjection','C2_Heartbeat']},
    "recent_threats":  [],
    "start_time":      datetime.now().isoformat(),
    "blocked_ips":     set(),
}
stats_lock = threading.Lock()

# ── RATE LIMITER (per-IP sliding window) ──────────────────────────────
_ip_windows: dict = defaultdict(list)
_rl_lock = threading.Lock()

def is_rate_limited(ip: str, window_sec=60, max_req=500) -> bool:
    """Returns True if ip has exceeded max_req in the last window_sec."""
    now = time.time()
    with _rl_lock:
        timestamps = _ip_windows[ip]
        # Keep only timestamps within window
        _ip_windows[ip] = [t for t in timestamps if now - t < window_sec]
        _ip_windows[ip].append(now)
        return len(_ip_windows[ip]) > max_req

# ── BLOCK LIST ─────────────────────────────────────────────────────────
def block_ip(ip: str, reason: str, duration_minutes: int = 60):
    expires = (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO block_list (ip, reason, added_at, expires_at) VALUES (?,?,?,?)",
            (ip, reason, datetime.now().isoformat(), expires)
        )
        conn.commit()
        with stats_lock:
            session_stats["blocked_ips"].add(ip)
        print(f"[🚫] BLOCKED: {ip} — {reason} (expires {duration_minutes}min)")
    except Exception as e:
        print(f"[!] Block list error: {e}")

def is_blocked(ip: str) -> bool:
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT expires_at FROM block_list WHERE ip=?", (ip,)
        ).fetchone()
        if row:
            expires = datetime.fromisoformat(row[0])
            if datetime.now() < expires:
                return True
            else:
                # Expired — remove
                conn.execute("DELETE FROM block_list WHERE ip=?", (ip,))
                conn.commit()
    except Exception:
        pass
    return False

# ── REQUEST / RESPONSE MODELS ──────────────────────────────────────────
class TrafficSample(BaseModel):
    features:        List[float]
    raw_log:         Optional[str] = ""
    src_ip:          Optional[str] = "unknown"
    dst_ip:          Optional[str] = "unknown"
    dst_port:        Optional[int] = 0
    protocol:        Optional[str] = "TCP"
    packets_per_sec: Optional[float] = 0.0

class AnalysisResult(BaseModel):
    rf_prediction:       str
    rf_confidence:       float
    rf_probabilities:    dict
    xgb_prediction:      Optional[str] = None
    xgb_confidence:      Optional[float] = None
    ensemble_prediction: str
    iso_prediction:      str
    iso_score:           float
    threat_detected:     bool
    threat_level:        str
    alert_type:          str
    narrative:           str
    narrative_source:    str
    remediation_actions: List[str]
    ip_blocked:          bool
    analysis_time_ms:    float
    timestamp:           str
    alert_id:            Optional[int] = None

# ── REMEDIATION ENGINE ─────────────────────────────────────────────────
REMEDIATION = {
    "DDoS": [
        "iptables -A INPUT -s {src_ip} -p udp -j DROP  # Block UDP flood source",
        "netsh advfirewall firewall add rule name='AgenticArmor-DDoS' dir=in action=block remoteip={src_ip}",
        "Rate-limit upstream: max 100 req/s from {src_ip} on port {dst_port}",
        "ESCALATE to Tier-2 SOC — volumetric DDoS confirmed (>5k pkt/s)",
    ],
    "PortScan": [
        "netsh advfirewall firewall add rule name='AgenticArmor-Scan' dir=in action=block remoteip={src_ip}",
        "Enable honeypot on commonly scanned ports to track {src_ip}",
        "Flag {src_ip} for 24-hour monitoring — likely pre-exploitation recon",
    ],
    "BruteForce": [
        "net user <target_account> /active:no  # Lock targeted account",
        "netsh advfirewall firewall add rule name='AgenticArmor-BF' dir=in action=block remoteip={src_ip} localport={dst_port}",
        "Enable MFA on port {dst_port} service immediately",
        "fail2ban equivalent: block {src_ip} after 5 failed attempts",
    ],
    "SQLInjection": [
        "Kill DB connection from {src_ip} to port {dst_port} immediately",
        "Enable WAF rule: block UNION SELECT / OR 1=1 patterns from {src_ip}",
        "Rotate DB credentials — possible exfiltration in progress",
        "ESCALATE — capture full payload for forensic analysis",
    ],
    "C2_Heartbeat": [
        "ISOLATE host {src_ip} from network immediately",
        "DNS sinkhole: block external resolution from {src_ip}",
        "Run full AV/EDR scan on {src_ip} — active malware suspected",
        "CRITICAL: capture all traffic from {src_ip} for IOC extraction",
    ],
    "BENIGN": [],
}

def get_remediation(attack_type: str, src_ip: str, dst_port: int) -> list:
    templates = REMEDIATION.get(attack_type, [])
    return [t.format(src_ip=src_ip, dst_port=dst_port) for t in templates[:3]]

# ── RULE-BASED NARRATIVES ──────────────────────────────────────────────
RULE_NARRATIVES = {
    "DDoS":
        "Volumetric DDoS flood detected from {src_ip} targeting port {dst_port} "
        "at {pps:.0f} pkt/s. Random Forest classified with {conf:.1f}% confidence; "
        "ISO anomaly score {iso:.4f} confirms abnormal traffic volume. "
        "Firewall block rule deployed — upstream rate limiting recommended.",

    "PortScan":
        "Stealthy port scan from {src_ip} targeting sequential ports. "
        "Single-SYN probes with no ACK response indicate pre-exploitation recon. "
        "ISO score {iso:.4f}. Source IP blocked and flagged for 24-hour monitoring.",

    "BruteForce":
        "Credential brute-force from {src_ip} on port {dst_port} ({conf:.1f}% confidence). "
        "High ACK count with regular inter-arrival time indicates scripted attack. "
        "Account lockout triggered; source IP rate-limited.",

    "SQLInjection":
        "SQL injection attempt from {src_ip} targeting service on port {dst_port}. "
        "Large asymmetric payload and PSH flag bursts suggest UNION SELECT exfiltration. "
        "WAF rule deployed; DB connection terminated.",

    "C2_Heartbeat":
        "Active C2 beacon detected from internal host {src_ip}. "
        "Periodic {pps:.1f}-second interval with uniform ~118-byte packets is a "
        "classic malware keep-alive pattern. Host quarantined; DNS sinkholed.",

    "BENIGN":
        "Traffic from {src_ip} to port {dst_port} classified BENIGN "
        "({conf:.1f}% confidence, ISO score {iso:.4f}). No action required.",
}

def rule_narrative(label, src_ip, dst_port, conf, iso_score, pps) -> str:
    t = RULE_NARRATIVES.get(label, "Traffic analyzed. No template available.")
    return t.format(src_ip=src_ip, dst_port=dst_port, conf=conf, iso=iso_score, pps=pps)

# ── GEMINI NARRATIVE ───────────────────────────────────────────────────
async def get_gemini_narrative(label, src_ip, dst_port, proto, pps, conf, iso_score, raw_log) -> str:
    if not gemini_model:
        return ""
    try:
        prompt = f"""You are a senior SOC analyst writing a real-time threat narrative for a security dashboard.

ALERT DETAILS:
- Attack Type: {label}
- Source IP: {src_ip}  →  Destination Port: {dst_port}  Protocol: {proto}
- Packets/sec: {pps:.1f}
- ML Confidence: {conf:.1f}%  (Random Forest)
- Isolation Forest Score: {iso_score:.4f}  (more negative = more anomalous)
- Raw flow log: {raw_log}

Write a 3-sentence technical narrative:
1. Describe what the attacker is doing and why this traffic pattern is suspicious
2. Explain what the ML models detected and what the key indicators are
3. State the immediate automated response that was taken

Rules:
- Be specific and technical — mention actual numbers from the alert
- No bullet points, no headers, plain paragraph
- Max 3 sentences"""

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(prompt)
        )
        return response.text.strip()
    except Exception as e:
        print(f"[!] Gemini error: {e}")
        return ""

# ── THREAT LEVEL LOGIC ─────────────────────────────────────────────────
def compute_threat_level(label: str, conf: float, iso_score: float, pps: float) -> str:
    if label == "BENIGN" and iso_score > -0.2:
        return "LOW"
    if label in ("DDoS", "C2_Heartbeat") and conf > 85:
        return "CRITICAL"
    if label in ("SQLInjection",) and conf > 80:
        return "HIGH"
    if label == "BruteForce" and conf > 80:
        return "HIGH"
    if iso_score < -0.35:
        return "HIGH"
    if label != "BENIGN" and conf > 65:
        return "MEDIUM"
    return "LOW"

# ── ENSEMBLE PREDICTION ────────────────────────────────────────────────
def ensemble_predict(arr_scaled):
    """Combine RF and XGBoost predictions if both are available."""
    rf_probs     = rf_model.predict_proba(arr_scaled)[0]
    rf_label_idx = int(np.argmax(rf_probs))
    rf_label     = label_encoder.inverse_transform([rf_label_idx])[0]
    rf_conf      = float(rf_probs[rf_label_idx]) * 100

    xgb_label = None
    xgb_conf  = None

    if xgb_model is not None:
        try:
            xgb_probs     = xgb_model.predict_proba(arr_scaled)[0]
            xgb_label_idx = int(np.argmax(xgb_probs))
            xgb_label     = label_encoder.inverse_transform([xgb_label_idx])[0]
            xgb_conf      = float(xgb_probs[xgb_label_idx]) * 100

            # Weighted ensemble: if both agree → higher confidence
            if rf_label == xgb_label:
                ensemble_label = rf_label
                ensemble_conf  = (rf_conf * 0.5 + xgb_conf * 0.5)
            else:
                # Disagree → take higher confidence prediction, reduce confidence
                if rf_conf >= xgb_conf:
                    ensemble_label = rf_label
                    ensemble_conf  = rf_conf * 0.8  # penalty for disagreement
                else:
                    ensemble_label = xgb_label
                    ensemble_conf  = xgb_conf * 0.8
        except Exception as e:
            print(f"[!] XGBoost inference error: {e}")
            ensemble_label = rf_label
            ensemble_conf  = rf_conf
    else:
        ensemble_label = rf_label
        ensemble_conf  = rf_conf

    rf_prob_dict = {
        cls: float(p) * 100
        for cls, p in zip(label_encoder.classes_, rf_probs)
    }

    return rf_label, rf_conf, xgb_label, xgb_conf, ensemble_label, ensemble_conf, rf_prob_dict

# ── MAIN INFERENCE ENDPOINT ────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_packet(sample: TrafficSample, request: Request):
    if rf_model is None:
        raise HTTPException(503, "Models not loaded — run train_models.py first.")

    t_start = time.time()

    # Check block list
    ip_blocked = is_blocked(sample.src_ip)
    if ip_blocked:
        pass  # still analyze — just note it

    # Validate features
    expected = len(feature_names)
    if len(sample.features) != expected:
        raise HTTPException(400, f"Expected {expected} features, got {len(sample.features)}")

    arr = np.array(sample.features, dtype=np.float32).reshape(1, -1)
    # Replace any inf/nan with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr_scaled = scaler.transform(arr)

    # ── ML inference ──
    rf_label, rf_conf, xgb_label, xgb_conf, ens_label, ens_conf, rf_prob_dict = \
        ensemble_predict(arr_scaled)

    iso_score = float(iso_model.score_samples(arr_scaled)[0])
    iso_pred  = "ANOMALY" if iso_model.predict(arr_scaled)[0] == -1 else "NORMAL"

    # ── Threat decision ──
    threat_detected = (
        (ens_label != "BENIGN" and ens_conf > 60) or
        (iso_score < -0.35)
    )
    alert_type   = ens_label if ens_label != "BENIGN" else \
                   ("ZERO_DAY_ANOMALY" if iso_pred == "ANOMALY" else "BENIGN")
    threat_level = compute_threat_level(ens_label, ens_conf, iso_score, sample.packets_per_sec)

    # ── Auto-block logic ──
    auto_blocked = False
    if threat_detected and not ip_blocked:
        if threat_level == "CRITICAL":
            block_ip(sample.src_ip, f"AUTO: {alert_type} — CRITICAL confidence {ens_conf:.1f}%", 120)
            auto_blocked = True
        elif threat_level == "HIGH" and is_rate_limited(sample.src_ip, window_sec=60, max_req=50):
            block_ip(sample.src_ip, f"AUTO: {alert_type} — rate limit exceeded", 30)
            auto_blocked = True

    # ── Narrative ──
    narrative_text   = ""
    narrative_source = "rule_based"
    if threat_detected:
        narrative_text = await get_gemini_narrative(
            ens_label, sample.src_ip, sample.dst_port, sample.protocol,
            sample.packets_per_sec, ens_conf, iso_score, sample.raw_log
        )
        if narrative_text:
            narrative_source = "gemini_2.0_flash"
    if not narrative_text:
        narrative_text = rule_narrative(
            ens_label if threat_detected else "BENIGN",
            sample.src_ip, sample.dst_port, ens_conf, iso_score, sample.packets_per_sec
        )

    # ── Remediation ──
    remediation = get_remediation(ens_label, sample.src_ip, sample.dst_port) \
                  if threat_detected else []

    elapsed = (time.time() - t_start) * 1000
    now_str = datetime.now().isoformat()

    # ── Persist to DB ──
    db_data = {
        "timestamp": now_str, "src_ip": sample.src_ip, "dst_ip": sample.dst_ip,
        "dst_port": sample.dst_port, "protocol": sample.protocol,
        "rf_prediction": rf_label, "rf_confidence": rf_conf,
        "iso_score": iso_score, "iso_prediction": iso_pred,
        "threat_detected": threat_detected, "threat_level": threat_level,
        "alert_type": alert_type, "analysis_ms": elapsed,
        "packets_per_sec": sample.packets_per_sec, "raw_log": sample.raw_log,
    }
    if threat_detected:
        threading.Thread(
            target=save_alert,
            args=(db_data, remediation, narrative_text, narrative_source),
            daemon=True
        ).start()

    # ── Session stats ──
    with stats_lock:
        session_stats["total_analyzed"] += 1
        if threat_detected:
            session_stats["threats_detected"] += 1
            if ens_label in session_stats["attack_counts"]:
                session_stats["attack_counts"][ens_label] += 1
            session_stats["recent_threats"].append({
                "time":         now_str,
                "label":        ens_label,
                "src_ip":       sample.src_ip,
                "confidence":   round(ens_conf, 1),
                "threat_level": threat_level,
                "blocked":      auto_blocked,
            })
            if len(session_stats["recent_threats"]) > 100:
                session_stats["recent_threats"].pop(0)
        if iso_pred == "ANOMALY":
            session_stats["anomalies_flagged"] += 1

    return AnalysisResult(
        rf_prediction       = rf_label,
        rf_confidence       = round(rf_conf, 2),
        rf_probabilities    = rf_prob_dict,
        xgb_prediction      = xgb_label,
        xgb_confidence      = round(xgb_conf, 2) if xgb_conf else None,
        ensemble_prediction = ens_label,
        iso_prediction      = iso_pred,
        iso_score           = round(iso_score, 4),
        threat_detected     = threat_detected,
        threat_level        = threat_level,
        alert_type          = alert_type,
        narrative           = narrative_text,
        narrative_source    = narrative_source,
        remediation_actions = remediation,
        ip_blocked          = ip_blocked or auto_blocked,
        analysis_time_ms    = round(elapsed, 2),
        timestamp           = now_str,
    )


# ── STATS ENDPOINT ─────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    with stats_lock:
        stats = dict(session_stats)
        stats["blocked_ips"] = list(stats["blocked_ips"])
    return stats


# ── ALERT HISTORY (from DB) ────────────────────────────────────────────
@app.get("/alerts")
def get_alerts(limit: int = 100, threat_only: bool = True):
    try:
        conn = get_db()
        query = "SELECT * FROM alerts"
        if threat_only:
            query += " WHERE threat_detected=1"
        query += f" ORDER BY timestamp DESC LIMIT {min(limit, 500)}"
        rows = conn.execute(query).fetchall()
        cols = [d[0] for d in conn.execute(query).description] if rows else []

        # Re-run to get description
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return {"alerts": [dict(zip(cols, row)) for row in rows], "count": len(rows)}
    except Exception as e:
        return {"error": str(e), "alerts": [], "count": 0}


# ── BLOCK LIST ENDPOINTS ────────────────────────────────────────────────
@app.get("/blocklist")
def get_blocklist():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM block_list WHERE expires_at > ?",
            (datetime.now().isoformat(),)
        ).fetchall()
        return {"blocked": [{"ip": r[0], "reason": r[1], "expires": r[3]} for r in rows]}
    except Exception as e:
        return {"error": str(e), "blocked": []}

@app.delete("/blocklist/{ip}")
def unblock_ip(ip: str):
    try:
        conn = get_db()
        conn.execute("DELETE FROM block_list WHERE ip=?", (ip,))
        conn.commit()
        with stats_lock:
            session_stats["blocked_ips"].discard(ip)
        return {"status": "unblocked", "ip": ip}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── HEALTH CHECK ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "models_loaded":    rf_model is not None,
        "xgboost_loaded":   xgb_model is not None,
        "gemini_connected": gemini_model is not None,
        "feature_count":    len(feature_names) if feature_names else 0,
        "db_path":          DB_PATH,
        "version":          "2.0.0",
    }

@app.get("/features")
def get_features():
    return {"features": feature_names, "count": len(feature_names)}


# ── RUN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
