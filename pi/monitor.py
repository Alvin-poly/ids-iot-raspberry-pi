import joblib
import numpy as np
import logging
import time
import subprocess
import pandas as pd
import re
from datetime import datetime

# ── Setup logging ──────────────────────────────────────────
logging.basicConfig(
    filename="/home/alvin09/ids/alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ── Load model and scaler ──────────────────────────────────
print("Loading model...")
model         = joblib.load("/home/alvin09/ids/ids_rf_model.pkl")
model.verbose = 0
scaler        = joblib.load("/home/alvin09/ids/scaler.pkl")
feature_names = joblib.load("/home/alvin09/ids/feature_names.pkl")
print(f"Model loaded. Expects {len(feature_names)} features.")
print("Starting live monitoring...\n")

# ── Feature extraction from tcpdump ───────────────────────
def capture_packets(interface="eth0", count=50):
    """Capture packets and return basic stats."""
    try:
        result = subprocess.run(
            ["sudo", "tcpdump", "-i", interface,
             "-c", str(count), "-q", "--immediate-mode"],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return ""

def extract_features_from_capture(raw_output):
    lines = [l for l in raw_output.strip().split("\n") if "IP" in l]

    if not lines:
        return None

    pkt_count = len(lines)
    ports = []
    protocols = []
    lengths = []

    for line in lines:
        port_match = re.search(r"\.(\d+)[\s>:]", line)
        if port_match:
            ports.append(int(port_match.group(1)))

        if "tcp" in line.lower():
            protocols.append(6)
        elif "udp" in line.lower():
            protocols.append(17)
        elif "icmp" in line.lower():
            protocols.append(1)
        else:
            protocols.append(0)

        len_match = re.search(r"length (\d+)", line)
        if len_match:
            lengths.append(int(len_match.group(1)))

    dst_port    = ports[-1] if ports else 0
    protocol    = protocols[-1] if protocols else 0
    avg_len     = sum(lengths) / len(lengths) if lengths else 0
    total_bytes = sum(lengths)

    # More realistic flow features
    flow_duration  = max(pkt_count * 100000, 1)
    fwd_pkts_s     = pkt_count / max(flow_duration / 1e6, 0.001)
    flow_byts_s    = total_bytes / max(flow_duration / 1e6, 0.001)

    base_features = [
        dst_port,           # Dst Port
        protocol,           # Protocol
        flow_duration,      # Flow Duration
        pkt_count,          # Tot Fwd Pkts
        0,                  # Tot Bwd Pkts
        total_bytes,        # TotLen Fwd Pkts
        0,                  # TotLen Bwd Pkts
        avg_len,            # Fwd Pkt Len Max
        avg_len,            # Fwd Pkt Len Min
        avg_len,            # Fwd Pkt Len Mean
        0,                  # Fwd Pkt Len Std
        0,                  # Bwd Pkt Len Max
        0,                  # Bwd Pkt Len Min
        0,                  # Bwd Pkt Len Mean
        0,                  # Bwd Pkt Len Std
        flow_byts_s,        # Flow Byts/s
        fwd_pkts_s,         # Flow Pkts/s
        0, 0, 0, 0,         # IAT features
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1 if protocol == 6 else 0,  # Fwd PSH Flags
        0, 0, 0,
        pkt_count * 20,     # Fwd Header Len
        0,                  # Bwd Header Len
        fwd_pkts_s,         # Fwd Pkts/s
        0,                  # Bwd Pkts/s
        avg_len,            # Pkt Len Min
        avg_len,            # Pkt Len Max
        avg_len,            # Pkt Len Mean
        0,                  # Pkt Len Std
        0,                  # Pkt Len Var
        1 if protocol == 6 else 0,  # FIN Flag Cnt
        1 if protocol == 6 else 0,  # SYN Flag Cnt
        0, 0, 0, 0, 0, 0,
        0,                  # Down/Up Ratio
        avg_len,            # Pkt Size Avg
    ]

    # Pad remaining with 0
    feature_vector = base_features + [0.0] * (len(feature_names) - len(base_features))
    return feature_vector[:len(feature_names)]

# ── Main monitoring loop ───────────────────────────────────
INTERFACE  = "wlan0"    # change to wlan0 if using WiFi
INTERVAL   = 5        # seconds between each capture

alert_count  = 0
sample_count = 0

while True:
    try:
        raw = capture_packets(interface=INTERFACE, count=10)

        if not raw.strip():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No packets captured, retrying...")
            time.sleep(INTERVAL)
            continue

        features = extract_features_from_capture(raw)

        if features is None:
            time.sleep(INTERVAL)
            continue

        scaled = scaler.transform(pd.DataFrame([features], columns=feature_names))
        pred     = model.predict(scaled)[0]
        proba    = model.predict_proba(scaled)[0][1]

        sample_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        if pred == 1:
            alert_count += 1
            msg = (f"[{timestamp}] ATTACK DETECTED "
                   f"| confidence: {round(proba * 100, 1)}%")
            print(msg)
            logging.info(msg)
        else:
            print(f"[{timestamp}] Benign traffic "
                  f"| confidence: {round((1 - proba) * 100, 1)}%"
                  f" | alerts so far: {alert_count}/{sample_count}")

        time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped.")
        print(f"Total samples : {sample_count}")
        print(f"Total alerts  : {alert_count}")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(INTERVAL)
