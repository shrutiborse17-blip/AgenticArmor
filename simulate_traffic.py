"""
AgenticArmor — Traffic Simulator (Demo Mode)
=============================================
Generates synthetic traffic and sends it to the backend.
This is DEMO MODE — no real packets are captured.

For REAL packet capture from your NIC, use:
    python capture.py

Usage:
    python simulate_traffic.py                    # mixed traffic
    python simulate_traffic.py --rate 0.5         # 2 packets/sec
    python simulate_traffic.py --attack DDoS      # force attack type
    python simulate_traffic.py --count 50         # send 50 then stop
"""

import argparse
import time
import random
import requests
import numpy as np
from datetime import datetime

API_URL    = "http://localhost:8000/analyze"
N_FEATURES = 78

INTERNAL_IPS = [f"192.168.1.{i}" for i in range(2, 30)]
EXTERNAL_IPS = [f"203.0.{random.randint(1,255)}.{random.randint(1,254)}" for _ in range(20)]
DST_IPS      = ["10.0.0.1", "10.0.0.2", "10.0.0.5", "192.168.10.50"]


def make_features(label):
    f = np.zeros(N_FEATURES)
    if label == 'BENIGN':
        f[0]=random.choice([80,443,53,123,8080]); f[1]=random.choice([6,17])
        f[2]=random.randint(1000,100000); f[3]=random.randint(1,50)
        f[4]=random.randint(1,50); f[5]=random.randint(100,5000)
        f[16]=random.uniform(10,500); f[49]=random.randint(0,2); f[52]=random.randint(0,10)
    elif label == 'DDoS':
        f[0]=random.choice([80,443,53]); f[1]=17; f[2]=random.randint(100,5000)
        f[3]=random.randint(3000,10000); f[4]=random.randint(0,5)
        f[5]=random.randint(100000,500000); f[16]=random.uniform(5000,15000)
        f[48]=random.randint(500,2000); f[7]=random.randint(40,60)
    elif label == 'PortScan':
        f[0]=random.randint(1,65535); f[1]=6; f[2]=random.randint(10,500)
        f[3]=random.randint(1,3); f[4]=random.randint(0,2)
        f[5]=random.randint(40,80); f[16]=random.uniform(1,50)
        f[48]=1; f[52]=0; f[45]=random.randint(0,2)
    elif label == 'BruteForce':
        f[0]=random.choice([22,21,3389,23]); f[1]=6; f[2]=random.randint(500,3000)
        f[3]=random.randint(10,100); f[4]=random.randint(10,100)
        f[5]=random.randint(1000,8000); f[16]=random.uniform(20,200)
        f[48]=random.randint(1,5); f[52]=random.randint(5,50); f[18]=random.uniform(100,500)
    elif label == 'SQLInjection':
        f[0]=random.choice([3306,5432,1433,80,443]); f[1]=6
        f[2]=random.randint(5000,50000); f[3]=random.randint(5,30); f[4]=random.randint(5,30)
        f[5]=random.randint(5000,50000); f[7]=random.randint(500,5000)
        f[51]=random.randint(5,20); f[16]=random.uniform(1,15)
    elif label == 'C2_Heartbeat':
        f[0]=random.choice([443,8443,4444,8080]); f[1]=6
        f[2]=random.randint(28000,32000); f[3]=random.randint(2,6); f[4]=random.randint(2,6)
        f[5]=random.randint(110,130); f[16]=random.uniform(0.1,0.5)
        f[18]=random.uniform(28000,32000)
    f += np.random.normal(0, 0.3, N_FEATURES)
    return np.clip(f, 0, None).tolist()


ATTACK_META = {
    "DDoS":         (EXTERNAL_IPS, [80,443,53],    "UDP"),
    "PortScan":     (EXTERNAL_IPS, list(range(1,1024)), "TCP"),
    "BruteForce":   (EXTERNAL_IPS, [22,3389,21],   "TCP"),
    "SQLInjection": (EXTERNAL_IPS, [3306,5432],    "TCP"),
    "C2_Heartbeat": (INTERNAL_IPS, [443,8443],     "HTTPS"),
}
ATTACK_TYPES = list(ATTACK_META.keys())


def send_packet(force_label=None):
    if force_label:
        label = force_label
    else:
        label = "BENIGN" if random.random() < 0.70 else random.choice(ATTACK_TYPES)

    features  = make_features(label)
    dst_ip    = random.choice(DST_IPS)

    if label == "BENIGN":
        src_ip   = random.choice(INTERNAL_IPS)
        dst_port = random.choice([80, 443, 53, 123])
        proto    = random.choice(["TCP","UDP"])
        pps      = random.uniform(10, 200)
    else:
        src_pool, ports, proto = ATTACK_META[label]
        src_ip   = random.choice(src_pool)
        dst_port = random.choice(ports)
        pps      = features[16]

    raw_log = f"{src_ip} → {dst_ip}:{dst_port} | {label} | {pps:.0f} pkt/s | {proto}"

    payload = {
        "features":        features,
        "raw_log":         raw_log,
        "src_ip":          src_ip,
        "dst_ip":          dst_ip,
        "dst_port":        int(dst_port),
        "protocol":        proto,
        "packets_per_sec": float(pps),
    }

    try:
        resp   = requests.post(API_URL, json=payload, timeout=10)
        result = resp.json()
        ts     = datetime.now().strftime("%H:%M:%S")

        pred   = result["rf_prediction"]
        conf   = result["rf_confidence"]
        iso    = result["iso_score"]
        threat = result["threat_detected"]
        level  = result["threat_level"]

        color = "\033[91m" if threat else "\033[92m"
        reset = "\033[0m"
        print(f"[{ts}] {color}{pred:15s}{reset} conf={conf:5.1f}%  "
              f"iso={iso:7.4f}  level={level:8s}  src={src_ip}")
        if threat:
            narrative = result.get("narrative", "")
            print(f"          ↳ {narrative[:110]}")
            for action in result.get("remediation_actions", [])[:1]:
                print(f"          ⚡ {action}")
        return result
    except requests.exceptions.ConnectionError:
        print("[!] Cannot connect. Is main.py running?")
    except Exception as e:
        print(f"[!] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="AgenticArmor Demo Traffic Simulator")
    parser.add_argument("--rate",   type=float, default=1.0,  help="Seconds between packets")
    parser.add_argument("--attack", type=str,   default=None, help="Force attack type")
    parser.add_argument("--count",  type=int,   default=0,    help="Packets to send (0=infinite)")
    args = parser.parse_args()

    print("=" * 65)
    print("  AgenticArmor — DEMO MODE (synthetic traffic)")
    print("  For real NIC capture, use: python capture.py")
    print(f"  Rate: 1 packet / {args.rate}s  |  Ctrl+C to stop")
    print("=" * 65 + "\n")

    i = 0
    while True:
        send_packet(force_label=args.attack)
        i += 1
        if args.count > 0 and i >= args.count:
            print(f"\n[✓] Sent {i} packets.")
            break
        time.sleep(args.rate)


if __name__ == "__main__":
    main()
