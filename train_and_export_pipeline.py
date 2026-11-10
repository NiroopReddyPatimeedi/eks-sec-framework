#!/usr/bin/env python3
"""
Retrain & Export Scikit-Learn Pipeline Artifact
----------------------------------------------
Bundles StandardScaler and IsolationForest into a single model.joblib artifact
and packages model.tar.gz for AWS SageMaker deployment.
"""

import os
import tarfile
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# 1. Generate baseline dataset matching original 6D feature distributions
np.random.seed(42)
n_normal = 172
n_admin = 125

# Normal Baseline Traffic: [process_count, syscall_rate, network_bytes, file_opens, cpu_delta, is_attack_probe]
normal_data = np.column_stack([
    np.random.uniform(5, 12, n_normal),
    np.random.uniform(100, 300, n_normal),
    np.random.uniform(200, 800, n_normal),
    np.random.uniform(8, 18, n_normal),
    np.random.uniform(2.20, 2.45, n_normal),
    np.zeros(n_normal)
])

# Authorized Admin Traffic
admin_data = np.column_stack([
    np.random.uniform(15, 25, n_admin),
    np.random.uniform(350, 600, n_admin),
    np.random.uniform(1000, 2500, n_admin),
    np.random.uniform(20, 35, n_admin),
    np.random.uniform(2.25, 2.48, n_admin),
    np.zeros(n_admin)
])

X_train = np.vstack([normal_data, admin_data])

# 2. Build Pipeline (Scaler + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(contamination=0.05, random_state=42, n_estimators=100))
])

# 3. Fit pipeline on raw telemetry data
pipeline.fit(X_train)

# 4. Save local joblib artifact
joblib.dump(pipeline, "model.joblib")
print("[✓] Saved pipeline to 'model.joblib'")

# 5. Archive into model.tar.gz for AWS SageMaker endpoint deployment
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model.joblib", arcname="model.joblib")
print("[✓] Packaged artifact into 'model.tar.gz'")