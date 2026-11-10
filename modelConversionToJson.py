#!/usr/bin/env python3
"""
Model Artifact Converter: joblib -> model.json
----------------------------------------------
Safely unwraps Scikit-Learn Pipeline or standalone IsolationForest 
artifacts and exports decision tree structures and scaler parameters to model.json.
"""

import json
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

MODEL_JOBLIB_PATH = "model.joblib"
MODEL_JSON_PATH = "model.json"

FEATURE_NAMES = [
    "process_spawn_rate",
    "memory_rss_kb",
    "cpu_pct",
    "syscall_rate",
    "net_io_kb",
    "active_sockets"
]

def convert_model_to_json():
    if not os.path.exists(MODEL_JOBLIB_PATH):
        raise FileNotFoundError(f"Missing required file: '{MODEL_JOBLIB_PATH}'")

    print(f"[*] Loading '{MODEL_JOBLIB_PATH}'...")
    loaded_artifact = joblib.load(MODEL_JOBLIB_PATH)

    # Unwrap IsolationForest and StandardScaler if encapsulated in Pipeline
    if isinstance(loaded_artifact, Pipeline) or hasattr(loaded_artifact, "named_steps"):
        print(" [✓] Identified Scikit-Learn Pipeline object.")
        iforest = loaded_artifact.named_steps.get("model", loaded_artifact.steps[-1][1])
        scaler = loaded_artifact.named_steps.get("scaler", None)
    elif isinstance(loaded_artifact, IsolationForest):
        print(" [*] Identified standalone IsolationForest object.")
        iforest = loaded_artifact
        scaler = None
    else:
        raise TypeError(f"Unrecognized model type: {type(loaded_artifact)}")

    print("[*] Extracting decision tree structures & metadata...")
    trees_metadata = []
    if hasattr(iforest, "estimators_"):
        for tree_idx, est in enumerate(iforest.estimators_):
            tree_struct = est.tree_
            trees_metadata.append({
                "tree_index": tree_idx,
                "node_count": int(tree_struct.node_count),
                "max_depth": int(est.max_depth) if est.max_depth is not None else -1,
                "thresholds": tree_struct.threshold.tolist()[:10]  # Top 10 node split thresholds
            })

    scaler_params = {}
    if scaler is not None and hasattr(scaler, "mean_"):
        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist()
        }

    model_metadata = {
        "model_type": "IsolationForest",
        "is_pipeline": isinstance(loaded_artifact, Pipeline),
        "n_estimators": getattr(iforest, "n_estimators", 100),
        "contamination": getattr(iforest, "contamination", 0.05),
        "feature_names": FEATURE_NAMES,
        "scaler_params": scaler_params,
        "tree_count": len(trees_metadata),
        "sample_tree_structures": trees_metadata[:5]
    }

    with open(MODEL_JSON_PATH, "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"[✓] Successfully exported metadata to '{MODEL_JSON_PATH}'")

if __name__ == "__main__":
    convert_model_to_json()