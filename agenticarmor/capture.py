"""
AgenticArmor — Live Packet Capture Engine (Windows)
=====================================================
Captures real network traffic from your NIC using Scapy,
extracts CIC-IDS-2018 compatible features from live flows,
and POSTs them to the FastAPI backend for real ML analysis.

This replaces simulate_traffic.py with REAL packet data.

Requirements:
  - Npcap installed: https://npcap.com/#download  (free, install with WinPcap compat mode)
  - pip install scapy
  - Run as Administrator (required for raw packet capture on Windows)

Usage:
    python capture.py                        # auto-select interface
    python capture.py --iface "Wi-Fi"        # specific interface
    python capture.py --iface "Ethernet"
    python capture.py --list-ifaces          # list available interfaces
    python capture.py --iface "Wi-Fi" --bpf "tcp"   # filter by BPF
    python capture.py --dry-run              # capture but don't send to API
"""

import argparse
import time
import threading
import json
import sys
import os
import math
import requests
import numpy as np
from collections import defaultdict
from datetime import datetime

# ── SCAPY IMPORT ──────────────────────────────────────────────────────────
try:
    from scapy.all import (
        sniff, get_if_list, get_if_hwaddr, IP, TCP, UDP, ICMP,
        conf as scapy_conf
    )
    from scapy.arch.windows import get_windows_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[!] Scapy not installed. Run: pip install scapy")
    print("[!] Also install Npcap from https://npcap.com/#download")

API_URL = "http://localhost:8000/analyze"

# Flow timeout — export flow after this many seconds of inactivity
FLOW_TIMEOUT = 5.0

# Minimum packets per flow before export (avoid single-packet noise)
MIN_PKTS = 3

# ── FLOW RECORD ────────────────────────────────────────────────────────────
class FlowRecord:
    """Tracks a single bidirectional network flow."""
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol

        self.start_time = None
        self.last_time  = None

        # Forward = src→dst, backward = dst→src
        self.fwd_pkts   = []   # list of (timestamp, length, flags)
        self.bwd_pkts   = []

        # Flags
        self.fin_cnt = self.syn_cnt = self.rst_cnt = 0
        self.psh_cnt = self.ack_cnt = self.urg_cnt = 0
        self.cwe_cnt = self.ece_cnt = 0

        self.init_fwd_win = -1
        self.init_bwd_win = -1

    def add_packet(self, ts, length, is_fwd, flags=0, win=0):
        if self.start_time is None:
            self.start_time = ts
        self.last_time = ts

        info = (ts, length, flags)
        if is_fwd:
            self.fwd_pkts.append(info)
            if self.init_fwd_win == -1:
                self.init_fwd_win = win
        else:
            self.bwd_pkts.append(info)
            if self.init_bwd_win == -1:
                self.init_bwd_win = win

        # Parse TCP flags
        if flags:
            if flags & 0x01: self.fin_cnt += 1
            if flags & 0x02: self.syn_cnt += 1
            if flags & 0x04: self.rst_cnt += 1
            if flags & 0x08: self.psh_cnt += 1
            if flags & 0x10: self.ack_cnt += 1
            if flags & 0x20: self.urg_cnt += 1
            if flags & 0x40: self.ece_cnt += 1
            if flags & 0x80: self.cwe_cnt += 1

    def duration_us(self):
        if self.start_time and self.last_time:
            return (self.last_time - self.start_time) * 1e6
        return 0

    def _iat_stats(self, pkts):
        """Inter-arrival time stats from list of (ts, len, flags)."""
        if len(pkts) < 2:
            return 0, 0, 0, 0, 0  # tot, mean, std, max, min
        timestamps = [p[0] for p in pkts]
        iats = [(timestamps[i+1] - timestamps[i]) * 1e6 for i in range(len(timestamps)-1)]
        if not iats:
            return 0, 0, 0, 0, 0
        tot  = sum(iats)
        mean = tot / len(iats)
        std  = math.sqrt(sum((x - mean)**2 for x in iats) / len(iats)) if len(iats) > 1 else 0
        return tot, mean, std, max(iats), min(iats)

    def _flow_iat_stats(self):
        """IAT across all packets (fwd+bwd combined, sorted by ts)."""
        all_pkts = sorted(self.fwd_pkts + self.bwd_pkts, key=lambda x: x[0])
        if len(all_pkts) < 2:
            return 0, 0, 0, 0
        timestamps = [p[0] for p in all_pkts]
        iats = [(timestamps[i+1] - timestamps[i]) * 1e6 for i in range(len(timestamps)-1)]
        mean = sum(iats) / len(iats)
        std  = math.sqrt(sum((x - mean)**2 for x in iats) / len(iats)) if len(iats) > 1 else 0
        return mean, std, max(iats), min(iats)

    def to_feature_vector(self):
        """Extract all 78 CIC-IDS-2018 features from this flow."""
        dur = self.duration_us()

        fwd_lens  = [p[1] for p in self.fwd_pkts]
        bwd_lens  = [p[1] for p in self.bwd_pkts]
        all_lens  = fwd_lens + bwd_lens

        tot_fwd   = len(self.fwd_pkts)
        tot_bwd   = len(self.bwd_pkts)
        totlen_fwd = sum(fwd_lens)
        totlen_bwd = sum(bwd_lens)

        fwd_max  = max(fwd_lens) if fwd_lens else 0
        fwd_min  = min(fwd_lens) if fwd_lens else 0
        fwd_mean = totlen_fwd / tot_fwd if tot_fwd else 0
        fwd_std  = math.sqrt(sum((x-fwd_mean)**2 for x in fwd_lens)/tot_fwd) if tot_fwd > 1 else 0

        bwd_max  = max(bwd_lens) if bwd_lens else 0
        bwd_min  = min(bwd_lens) if bwd_lens else 0
        bwd_mean = totlen_bwd / tot_bwd if tot_bwd else 0
        bwd_std  = math.sqrt(sum((x-bwd_mean)**2 for x in bwd_lens)/tot_bwd) if tot_bwd > 1 else 0

        dur_s    = dur / 1e6 if dur > 0 else 1e-6
        flow_byts_s = (totlen_fwd + totlen_bwd) / dur_s
        flow_pkts_s = (tot_fwd + tot_bwd) / dur_s
        fwd_pkts_s  = tot_fwd / dur_s
        bwd_pkts_s  = tot_bwd / dur_s

        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._flow_iat_stats()

        fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._iat_stats(self.fwd_pkts)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._iat_stats(self.bwd_pkts)

        pkt_len_min  = min(all_lens) if all_lens else 0
        pkt_len_max  = max(all_lens) if all_lens else 0
        pkt_len_mean = sum(all_lens) / len(all_lens) if all_lens else 0
        pkt_len_std  = math.sqrt(sum((x-pkt_len_mean)**2 for x in all_lens)/len(all_lens)) if len(all_lens) > 1 else 0
        pkt_len_var  = pkt_len_std ** 2

        down_up = totlen_bwd / totlen_fwd if totlen_fwd > 0 else 0
        pkt_size_avg = pkt_len_mean
        fwd_seg_avg  = fwd_mean
        bwd_seg_avg  = bwd_mean

        proto_num = 6 if self.protocol == 'TCP' else (17 if self.protocol == 'UDP' else 0)

        # Build the 78-feature vector (order must match FEATURE_NAMES in train_models.py)
        f = [
            self.dst_port,          # 0  dst_port
            proto_num,              # 1  protocol
            dur,                    # 2  flow_duration
            tot_fwd,                # 3  tot_fwd_pkts
            tot_bwd,                # 4  tot_bwd_pkts
            totlen_fwd,             # 5  totlen_fwd_pkts
            totlen_bwd,             # 6  totlen_bwd_pkts
            fwd_max,                # 7  fwd_pkt_len_max
            fwd_min,                # 8  fwd_pkt_len_min
            fwd_mean,               # 9  fwd_pkt_len_mean
            fwd_std,                # 10 fwd_pkt_len_std
            bwd_max,                # 11 bwd_pkt_len_max
            bwd_min,                # 12 bwd_pkt_len_min
            bwd_mean,               # 13 bwd_pkt_len_mean
            bwd_std,                # 14 bwd_pkt_len_std
            flow_byts_s,            # 15 flow_byts_s
            flow_pkts_s,            # 16 flow_pkts_s
            flow_iat_mean,          # 17 flow_iat_mean
            flow_iat_std,           # 18 flow_iat_std
            flow_iat_max,           # 19 flow_iat_max
            flow_iat_min,           # 20 flow_iat_min
            fwd_iat_tot,            # 21 fwd_iat_tot
            fwd_iat_mean,           # 22 fwd_iat_mean
            fwd_iat_std,            # 23 fwd_iat_std
            fwd_iat_max,            # 24 fwd_iat_max
            fwd_iat_min,            # 25 fwd_iat_min
            bwd_iat_tot,            # 26 bwd_iat_tot
            bwd_iat_mean,           # 27 bwd_iat_mean
            bwd_iat_std,            # 28 bwd_iat_std
            bwd_iat_max,            # 29 bwd_iat_max
            bwd_iat_min,            # 30 bwd_iat_min
            0,                      # 31 fwd_psh_flags (per-flow, not per-pkt)
            0,                      # 32 bwd_psh_flags
            0,                      # 33 fwd_urg_flags
            0,                      # 34 bwd_urg_flags
            tot_fwd * 20,           # 35 fwd_header_len (approx)
            tot_bwd * 20,           # 36 bwd_header_len
            fwd_pkts_s,             # 37 fwd_pkts_s
            bwd_pkts_s,             # 38 bwd_pkts_s
            pkt_len_min,            # 39 pkt_len_min
            pkt_len_max,            # 40 pkt_len_max
            pkt_len_mean,           # 41 pkt_len_mean
            pkt_len_std,            # 42 pkt_len_std
            pkt_len_var,            # 43 pkt_len_var
            self.fin_cnt,           # 44 fin_flag_cnt
            self.syn_cnt,           # 45 syn_flag_cnt
            self.rst_cnt,           # 46 rst_flag_cnt
            self.psh_cnt,           # 47 psh_flag_cnt
            self.ack_cnt,           # 48 ack_flag_cnt
            self.urg_cnt,           # 49 urg_flag_cnt
            self.cwe_cnt,           # 50 cwe_flag_count
            self.ece_cnt,           # 51 ece_flag_cnt
            down_up,                # 52 down_up_ratio
            pkt_size_avg,           # 53 pkt_size_avg
            fwd_seg_avg,            # 54 fwd_seg_size_avg
            bwd_seg_avg,            # 55 bwd_seg_size_avg
            0,                      # 56 fwd_byts_b_avg
            0,                      # 57 fwd_pkts_b_avg
            0,                      # 58 fwd_blk_rate_avg
            0,                      # 59 bwd_byts_b_avg
            0,                      # 60 bwd_pkts_b_avg
            0,                      # 61 bwd_blk_rate_avg
            tot_fwd,                # 62 subflow_fwd_pkts
            totlen_fwd,             # 63 subflow_fwd_byts
            tot_bwd,                # 64 subflow_bwd_pkts
            totlen_bwd,             # 65 subflow_bwd_byts
            self.init_fwd_win if self.init_fwd_win > 0 else 0,  # 66 init_fwd_win_byts
            self.init_bwd_win if self.init_bwd_win > 0 else 0,  # 67 init_bwd_win_byts
            max(0, tot_fwd - 1),    # 68 fwd_act_data_pkts
            fwd_min,                # 69 fwd_seg_size_min
            0,                      # 70 active_mean  (requires idle detection)
            0,                      # 71 active_std
            0,                      # 72 active_max
            0,                      # 73 active_min
            0,                      # 74 idle_mean
            0,                      # 75 idle_std
            0,                      # 76 idle_max
            0,                      # 77 idle_min
        ]

        # Clamp all values — no infinities or NaNs
        f = [min(max(float(x), 0.0), 1e9) if math.isfinite(float(x)) else 0.0 for x in f]
        return f


# ── FLOW TABLE ─────────────────────────────────────────────────────────────
class FlowTable:
    def __init__(self, api_url, dry_run=False):
        self.flows    = {}
        self.lock     = threading.Lock()
        self.api_url  = api_url
        self.dry_run  = dry_run
        self.exported = 0
        self.threats  = 0

    def _flow_key(self, src_ip, dst_ip, src_port, dst_port, proto):
        """Bidirectional flow key."""
        a = (src_ip, src_port)
        b = (dst_ip, dst_port)
        if a > b:
            a, b = b, a
        return (*a, *b, proto)

    def add_packet(self, pkt):
        if not pkt.haslayer(IP):
            return
        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        ts     = float(pkt.time)
        length = len(pkt)

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            proto    = 'TCP'
            flags    = int(tcp.flags)
            win      = tcp.window
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport
            proto    = 'UDP'
            flags    = 0
            win      = 0
        else:
            return  # skip ICMP etc for now

        key = self._flow_key(src_ip, dst_ip, src_port, dst_port, proto)
        is_fwd = (src_ip, src_port) <= (dst_ip, dst_port)

        with self.lock:
            if key not in self.flows:
                self.flows[key] = FlowRecord(src_ip, dst_ip, src_port, dst_port, proto)
            self.flows[key].add_packet(ts, length, is_fwd, flags, win)

    def export_finished_flows(self):
        """Export flows that have timed out."""
        now = time.time()
        to_export = []
        with self.lock:
            for key, flow in list(self.flows.items()):
                if flow.last_time and (now - flow.last_time) > FLOW_TIMEOUT:
                    to_export.append(flow)
                    del self.flows[key]

        for flow in to_export:
            if (len(flow.fwd_pkts) + len(flow.bwd_pkts)) < MIN_PKTS:
                continue
            self._send_flow(flow)

    def _send_flow(self, flow):
        features = flow.to_feature_vector()
        proto_str = flow.protocol
        pps = features[16]  # flow_pkts_s

        raw_log = (
            f"{flow.src_ip}:{flow.src_port} → "
            f"{flow.dst_ip}:{flow.dst_port} | "
            f"{proto_str} | "
            f"{len(flow.fwd_pkts)+len(flow.bwd_pkts)} pkts | "
            f"{pps:.1f} pkt/s | "
            f"SYN={flow.syn_cnt} ACK={flow.ack_cnt} RST={flow.rst_cnt}"
        )

        payload = {
            "features":        features,
            "raw_log":         raw_log,
            "src_ip":          flow.src_ip,
            "dst_ip":          flow.dst_ip,
            "dst_port":        flow.dst_port,
            "protocol":        proto_str,
            "packets_per_sec": pps,
        }

        self.exported += 1
        ts = datetime.now().strftime("%H:%M:%S")

        if self.dry_run:
            print(f"[{ts}] [DRY-RUN] {raw_log}")
            return

        try:
            resp   = requests.post(self.api_url, json=payload, timeout=5)
            result = resp.json()
            label  = result["rf_prediction"]
            conf   = result["rf_confidence"]
            threat = result["threat_detected"]
            level  = result["threat_level"]

            color = "\033[91m" if threat else "\033[92m"
            reset = "\033[0m"

            print(f"[{ts}] {color}{label:15s}{reset} "
                  f"conf={conf:5.1f}% "
                  f"threat={str(threat):5s} "
                  f"level={level:8s} | {raw_log}")

            if threat:
                self.threats += 1
                narr = result.get("narrative", "")
                if narr:
                    print(f"         ↳ {narr[:120]}")
                for action in result.get("remediation_actions", [])[:2]:
                    print(f"         ↳ ACTION: {action}")

        except requests.exceptions.ConnectionError:
            print(f"[{ts}] [!] Cannot reach API at {self.api_url} — is main.py running?")
        except Exception as e:
            print(f"[{ts}] [!] Error sending flow: {e}")


# ── INTERFACE LISTING ───────────────────────────────────────────────────────
def list_interfaces():
    print("\nAvailable network interfaces:")
    print("-" * 50)
    try:
        ifaces = get_windows_if_list()
        for i, iface in enumerate(ifaces):
            name = iface.get('name', 'unknown')
            desc = iface.get('description', '')
            ips  = iface.get('ips', [])
            print(f"  [{i}] {name}")
            if desc:
                print(f"       Description: {desc}")
            if ips:
                print(f"       IPs: {', '.join(ips)}")
        print()
    except Exception:
        # Fallback
        for iface in get_if_list():
            print(f"  {iface}")
    print("Use: python capture.py --iface \"<name>\"")


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AgenticArmor Live Capture")
    parser.add_argument("--iface",       type=str,  default=None,
                        help="Network interface name (e.g. 'Wi-Fi', 'Ethernet')")
    parser.add_argument("--bpf",         type=str,  default="ip",
                        help="BPF filter string (default: 'ip')")
    parser.add_argument("--timeout",     type=float, default=FLOW_TIMEOUT,
                        help="Flow inactivity timeout in seconds (default: 5)")
    parser.add_argument("--list-ifaces", action="store_true",
                        help="List available interfaces and exit")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Capture and extract features but don't send to API")
    parser.add_argument("--api",         type=str,  default=API_URL,
                        help=f"Backend API URL (default: {API_URL})")
    args = parser.parse_args()

    if not SCAPY_AVAILABLE:
        print("[!] Scapy is required. Run: pip install scapy")
        print("[!] Also install Npcap from https://npcap.com/#download")
        sys.exit(1)

    if args.list_ifaces:
        list_interfaces()
        sys.exit(0)

    print("=" * 65)
    print("  AgenticArmor — Live Packet Capture Engine")
    print(f"  Interface : {args.iface or 'auto'}")
    print(f"  BPF Filter: {args.bpf}")
    print(f"  API       : {args.api}")
    print(f"  Dry Run   : {args.dry_run}")
    print("  Press Ctrl+C to stop")
    print("=" * 65)
    print()

    if not args.dry_run:
        print("[*] Checking API connection...")
        try:
            r = requests.get(args.api.replace("/analyze", "/health"), timeout=3)
            health = r.json()
            if health.get("models_loaded"):
                print(f"[✓] API connected. Models loaded. Features: {health.get('feature_count')}")
            else:
                print("[!] API reachable but models not loaded. Run train_models.py first.")
        except Exception:
            print(f"[!] Cannot reach API at {args.api.replace('/analyze', '/health')}")
            print("    Start the backend first: python main.py")
            sys.exit(1)

    flow_table = FlowTable(api_url=args.api, dry_run=args.dry_run)

    # Background thread: export finished flows every second
    def exporter():
        while True:
            time.sleep(1)
            flow_table.export_finished_flows()

    t = threading.Thread(target=exporter, daemon=True)
    t.start()

    print(f"[*] Starting capture on interface: {args.iface or 'default'}")
    print(f"[*] Active flows will be exported after {args.timeout}s of inactivity\n")

    try:
        sniff(
            iface=args.iface,
            filter=args.bpf,
            prn=flow_table.add_packet,
            store=False,
        )
    except KeyboardInterrupt:
        print(f"\n[✓] Capture stopped.")
        print(f"    Flows exported : {flow_table.exported}")
        print(f"    Threats found  : {flow_table.threats}")
    except Exception as e:
        print(f"[!] Capture error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Run as Administrator (right-click → Run as administrator)")
        print("  2. Install Npcap from https://npcap.com/#download")
        print("     → Check 'Install Npcap in WinPcap API-compatible Mode'")
        print("  3. Use --list-ifaces to see available interfaces")
        print("  4. Try --iface 'Wi-Fi' or --iface 'Ethernet'")


if __name__ == "__main__":
    main()
