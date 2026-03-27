"""
AgenticArmor — Test Traffic Simulator
=======================================
USE THIS ONLY FOR TESTING / DEMOS.
For real traffic analysis, use capture.py instead.

This sends synthetic feature vectors to the backend to verify
that the ML pipeline, Gemini narratives, and dashboard all work
before you go live with real packet capture.

Usage:
    python simulate_traffic.py                         # mixed traffic
    python simulate_traffic.py --attack DDoS           # force attack type
    python simulate_traffic.py --rate 0.5 --count 50  # 2/sec, 50 packets
"""

import argparse
import time
import random
import requests
import numpy as np
from datetime import datetime

API_URL = "http://localhost:8000/analyze"
N_FEATURES = 78

INTERNAL_IPS = [f"192.168.1.{i}" for i in range(2, 30)]
EXTERNAL_IPS = [f"203.{random.randint(0,255)}.{random.randint(1,255)}.{random.randint(1,254)}"
                for _ in range(30)]
DST_IPS      = ["10.0.0.1", "10.0.0.2", "10.0.0.5", "192.168.10.50"]


# ── FEATURE GENERATORS ────────────────────────────────────────────────
def _noise(f, scale=0.5):
    f = np.array(f, dtype=float)
    f += np.random.normal(0, scale, len(f))
    return np.clip(f, 0, None).tolist()

def make_benign():
    f = np.zeros(N_FEATURES)
    f[0]  = random.choice([80, 443, 53, 123, 8080])
    f[1]  = random.choice([6, 17])
    f[2]  = random.randint(1000, 100000)
    f[3]  = random.randint(2, 60)
    f[4]  = random.randint(2, 60)
    f[5]  = random.randint(200, 8000)
    f[15] = random.uniform(500, 5000)
    f[16] = random.uniform(5, 300)
    f[45] = random.randint(0, 2)
    f[48] = random.randint(2, 20)
    return _noise(f, 0.3)

def make_ddos():
    f = np.zeros(N_FEATURES)
    f[0]  = random.choice([80, 443, 53])
    f[1]  = 17
    f[2]  = random.randint(50, 3000)
    f[3]  = random.randint(5000, 40000)
    f[4]  = random.randint(0, 3)
    f[5]  = random.randint(200000, 1500000)
    f[7]  = random.randint(40, 65)
    f[15] = random.uniform(100000, 3000000)
    f[16] = random.uniform(5000, 40000)
    f[45] = random.randint(3000, 25000)
    return _noise(f, 0.5)

def make_portscan():
    f = np.zeros(N_FEATURES)
    f[0]  = random.randint(1, 65535)
    f[1]  = 6
    f[2]  = random.randint(5, 200)
    f[3]  = random.randint(1, 3)
    f[4]  = random.randint(0, 1)
    f[5]  = random.randint(40, 80)
    f[16] = random.uniform(5, 80)
    f[45] = 1
    f[48] = 0
    f[46] = random.randint(0, 1)
    return _noise(f, 0.2)

def make_brute():
    f = np.zeros(N_FEATURES)
    f[0]  = random.choice([22, 21, 3389, 23])
    f[1]  = 6
    f[2]  = random.randint(200, 5000)
    f[3]  = random.randint(15, 120)
    f[4]  = random.randint(15, 120)
    f[5]  = random.randint(2000, 12000)
    f[16] = random.uniform(10, 150)
    f[17] = random.uniform(50, 300)
    f[18] = random.uniform(5, 30)
    f[45] = random.randint(1, 5)
    f[48] = random.randint(10, 80)
    return _noise(f, 0.3)

def make_sqli():
    f = np.zeros(N_FEATURES)
    f[0]  = random.choice([3306, 5432, 1433, 80, 443])
    f[1]  = 6
    f[2]  = random.randint(5000, 80000)
    f[3]  = random.randint(5, 35)
    f[4]  = random.randint(5, 35)
    f[5]  = random.randint(15000, 100000)
    f[7]  = random.randint(1000, 8000)
    f[16] = random.uniform(1, 25)
    f[47] = random.randint(5, 25)
    return _noise(f, 0.3)

def make_c2():
    f = np.zeros(N_FEATURES)
    f[0]  = random.choice([443, 8443, 4444, 8080])
    f[1]  = 6
    f[2]  = random.randint(27000, 33000)
    f[3]  = random.randint(2, 6)
    f[4]  = random.randint(2, 6)
    f[5]  = random.randint(110, 130)
    f[7]  = random.randint(115, 125)
    f[16] = random.uniform(0.05, 0.3)
    f[17] = random.uniform(28000, 32000)
    f[18] = random.uniform(10, 100)
    return _noise(f, 0.2)


GENERATORS = {
    "DDoS":         (make_ddos,     EXTERNAL_IPS, [80,443,53],      "UDP"),
    "PortScan":     (make_portscan, EXTERNAL_IPS, list(range(1,1024)), "TCP"),
    "BruteForce":   (make_brute,    EXTERNAL_IPS, [22,3389,21],     "TCP"),
    "SQLInjection": (make_sqli,     EXTERNAL_IPS, [3306,5432,80],   "TCP"),
    "C2_Heartbeat": (make_c2,       INTERNAL_IPS, [443,8443],       "HTTPS"),
}


def send(attack_type=None):
    if attack_type is None:
        attack_type = "BENIGN" if random.random() < 0.65 else random.choice(list(GENERATORS))

    if attack_type == "BENIGN":
        features = make_benign()
        src_ip   = random.choice(INTERNAL_IPS)
        dst_ip   = random.choice(DST_IPS)
        dst_port = random.choice([80, 443, 53, 123])
        proto    = random.choice(["TCP", "UDP"])
        pps      = features[16]
    else:
        fn, src_pool, ports, proto = GENERATORS[attack_type]
        features = fn()
        src_ip   = random.choice(src_pool)
        dst_ip   = random.choice(DST_IPS)
        dst_port = random.choice(ports)
        pps      = features[16]

    raw_log = f"[TEST] {src_ip} → {dst_ip}:{dst_port} | {attack_type} | {pps:.1f} pkt/s | {proto}"

    payload = {
        "features":         features,
        "raw_log":          raw_log,
        "src_ip":           src_ip,
        "dst_ip":           dst_ip,
        "dst_port":         int(dst_port),
        "protocol":         proto,
        "packets_per_sec":  float(pps),
    }

    try:
        resp   = requests.post(API_URL, json=payload, timeout=10)
        result = resp.json()
        ts     = datetime.now().strftime("%H:%M:%S")

        label   = result.get("ensemble_prediction", result.get("rf_prediction","?"))
        conf    = result.get("rf_confidence", 0)
        iso     = result.get("iso_score", 0)
        threat  = result.get("threat_detected", False)
        level   = result.get("threat_level", "?")
        blocked = result.get("ip_blocked", False)

        color = "\033[91m" if threat else "\033[92m"
        reset = "\033[0m"
        block_tag = " [BLOCKED]" if blocked else ""

        print(f"[{ts}] {color}{label:15s}{reset} "
              f"conf={conf:5.1f}% iso={iso:7.4f} "
              f"level={level:8s} src={src_ip}{block_tag}")

        if threat:
            narr = result.get("narrative","")
            if narr:
                print(f"         ↳ {narr[:120]}")
            for action in result.get("remediation_actions", [])[:1]:
                print(f"         ↳ ACTION: {action}")

        return result

    except requests.exceptions.ConnectionError:
        print(f"[!] Cannot connect to {API_URL} — is main.py running?")
        return None
    except Exception as e:
        print(f"[!] Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="AgenticArmor Test Simulator")
    parser.add_argument("--rate",   type=float, default=1.0,
                        help="Seconds between packets (default: 1.0)")
    parser.add_argument("--attack", type=str,   default=None,
                        help="Force attack type: DDoS|PortScan|BruteForce|SQLInjection|C2_Heartbeat")
    parser.add_argument("--count",  type=int,   default=0,
                        help="Packet count (0 = infinite)")
    args = parser.parse_args()

    print("=" * 70)
    print("  AgenticArmor — TEST MODE (Synthetic Data)")
    print("  For real traffic, use: python capture.py --iface 'Wi-Fi'")
    print(f"  API: {API_URL}  |  Rate: 1 packet / {args.rate}s")
    print("  Ctrl+C to stop")
    print("=" * 70 + "\n")

    i = 0
    while True:
        send(attack_type=args.attack)
        i += 1
        if args.count > 0 and i >= args.count:
            print(f"\n[✓] Sent {i} packets. Done.")
            break
        time.sleep(args.rate)


if __name__ == "__main__":
    main()
