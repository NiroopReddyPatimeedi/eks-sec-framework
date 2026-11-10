#!/usr/bin/env python3
"""
Local Model Diagnostic - Multi-Cluster PCA Vector Evaluation
------------------------------------------------------------
Evaluates model.joblib against 6D telemetry vectors mapped directly 
from the PCA 2D Projection cluster boundaries (Normal, Admin, & 3 MITRE Sub-clusters).
"""

import os
import tarfile
import joblib
import numpy as np

MODEL_FILE = "model.joblib"
TAR_FILE = "model.tar.gz"

# Auto-extract model.joblib if compressed
if not os.path.exists(MODEL_FILE):
    if os.path.exists(TAR_FILE):
        print(f"[*] Extracting '{MODEL_FILE}' from '{TAR_FILE}'...")
        with tarfile.open(TAR_FILE, "r:gz") as tar:
            tar.extractall()
    else:
        raise FileNotFoundError(f"Neither '{MODEL_FILE}' nor '{TAR_FILE}' found.")

model = joblib.load(MODEL_FILE)
print(f"[+] Loaded model artifact: {MODEL_FILE}\n")

# Vector Schema: [process_count, syscall_rate, network_bytes, file_opens, cpu_delta, is_attack_probe]
PCA_ALIGNED_VECTORS = {
    "1. Normal Baseline (Green Cluster)": {
        "vector": np.array([[8, 200, 500, 12, 2.30, 0.0]]),
        "expected": "Normal"
    },
    "2. Authorized Admin (Blue Cluster)": {
        "vector": np.array([[22, 480, 1800, 28, 2.38, 0.0]]),
        "expected": "Normal"
    },
    "3. MITRE T1611 Escape (Red Top Cluster: PCA1 ~2.5, PCA2 ~4.0)": {
        "vector": np.array([[350, 28000, 120000, 900, 5.80, 1.0]]),
        "expected": "Anomaly"
    },
    "4. MITRE T1059 Exec (Red Mid-Right Cluster: PCA1 ~3.5, PCA2 ~1.8)": {
        "vector": np.array([[180, 15000, 35000, 420, 5.10, 1.0]]),
        "expected": "Anomaly"
    },
    "5. MITRE T1041 Exfil (Red Bottom-Right Cluster: PCA1 ~6.0, PCA2 ~ -3.0)": {
        "vector": np.array([[110, 12000, 750000, 310, 4.90, 1.0]]),
        "expected": "Anomaly"
    }
}

print("==================================================================")
print("       LOCAL ISOLATION FOREST TEST: PCA CLUSTERS EVALUATION       ")
print("==================================================================")

for label, data in PCA_ALIGNED_VECTORS.items():
    vector = data["vector"]
    pred = model.predict(vector)[0]  # 1 = Normal, -1 = Anomaly
    score = model.decision_function(vector)[0]
    
    predicted_label = "Normal" if pred == 1 else "Anomaly"
    status = "PASSED" if predicted_label == data["expected"] else "FAILED"
    
    print(f"\nCluster Name   : {label}")
    print(f"Feature Vector : {vector.tolist()[0]}")
    print(f"Raw Predict    : {pred} ({predicted_label})")
    print(f"Anomaly Score  : {score:.4f}")
    print(f"Test Verdict   : [{status}]")

print("==================================================================")