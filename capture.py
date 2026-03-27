from scapy.all import sniff, IP, TCP, UDP
import requests
import time

API_URL = "http://127.0.0.1:8000/analyze"

# ✅ IMPORTANT: your model expects 78 features
FEATURE_COUNT = 78

flows = {}

def process_packet(packet):
    try:
        if not packet.haslayer(IP):
            return
        
        src = packet[IP].src
        dst = packet[IP].dst
        
        proto = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "OTHER"
        
        sport = packet.sport if hasattr(packet, 'sport') else 0
        dport = packet.dport if hasattr(packet, 'dport') else 0

        key = (src, dst, sport, dport)

        # Initialize flow
        if key not in flows:
            flows[key] = {
                "packets": 0,
                "bytes": 0,
                "start_time": time.time()
            }

        # Update flow stats
        flows[key]["packets"] += 1
        flows[key]["bytes"] += len(packet)

        duration = time.time() - flows[key]["start_time"]

        packet_count = flows[key]["packets"]
        byte_count = flows[key]["bytes"]
        packets_per_sec = packet_count / max(duration, 1)

        # 🎯 BASIC REAL FEATURES (rest will be padded)
        features = [
            dport,                 # dst_port
            1 if proto == "TCP" else 2 if proto == "UDP" else 0,  # protocol encoding
            duration,              # flow_duration
            packet_count,          # tot_fwd_pkts (approx)
            0,                     # tot_bwd_pkts (not tracked)
            byte_count,            # totlen_fwd_pkts
            0                      # totlen_bwd_pkts
        ]

        # 🔧 Pad remaining features to 78
        while len(features) < FEATURE_COUNT:
            features.append(0)

        payload = {
            "features": features,
            "src_ip": src,
            "dst_ip": dst,
            "src_port": sport,
            "dst_port": dport,
            "protocol": proto,
            "packets_per_sec": packets_per_sec,
            "raw_log": str(packet.summary())
        }

        # 🚨 Send every 5 packets (flow-based)
        if packet_count % 5 == 0:
            response = requests.post(API_URL, json=payload)

            print("\n🚨 FLOW DETECTED")
            print(f"SRC: {src} → DST: {dst}")
            print(f"Protocol: {proto} | Port: {dport}")
            print(f"Packets: {packet_count} | Bytes: {byte_count}")
            print("📊 Response:", response.json())

    except Exception as e:
        print("Error:", e)

print("[*] REAL flow-based capture started...")

# Start sniffing
sniff(prn=process_packet, store=False)
