#!/usr/bin/env python3
"""
Single-Vector SageMaker Inference Diagnostic
-------------------------------------------
Dispatches standalone feature vectors for Normal, Admin, and MITRE Attack 
traffic to verify SageMaker Isolation Forest decision outputs directly.
"""

import json
import logging
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "eks-isolation-forest-endpoint")

# Configure ASCII-safe logging for Windows Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SageMaker-Diagnostic")

session = boto3.Session(region_name=AWS_REGION)
sm_runtime = session.client("sagemaker-runtime")

# Calibrated 6D Feature Vectors:
# [process_count, syscall_rate, network_bytes, file_opens, cpu_delta, is_attack_probe]
TEST_CASES = [
    {
        "type": "NORMAL",
        "vector": [8, 200, 500, 12, 2.30, 0.0],
        "description": "Standard user traffic (Low process count, cpu_delta=2.30, probe=0.0)"
    },
    {
        "type": "ADMIN",
        "vector": [20, 450, 1500, 25, 2.35, 0.0],
        "description": "Authorized admin query (Moderate process count, cpu_delta=2.35, probe=0.0)"
    },
    {
        "type": "ATTACK",
        "vector": [60, 1800, 10000, 75, 5.20, 1.0],
        "description": "MITRE exploit vector (High syscall rate, cpu_delta=5.20, probe=1.0)"
    }
]


def test_single_vector(case: dict):
    logger.info(f"--- TEST CASE: {case['type']} ---")
    logger.info(f"Description : {case['description']}")
    logger.info(f"Vector Sent : {case['vector']}")

    payload_json = json.dumps({"features": case["vector"]})
    start_time = time.perf_counter()

    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload_json
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        raw_body = response["Body"].read().decode("utf-8")
        result = json.loads(raw_body)

        anomaly_score = result.get("anomaly_score", 0.0)
        is_anomaly = result.get("is_anomaly", 0)

        logger.info(f"Inference Latency: {latency_ms:.2f} ms")
        logger.info(f"Anomaly Score    : {anomaly_score:.4f}")
        logger.info(f"Is Anomaly Flag  : {is_anomaly}")
        logger.info(f"Raw Output       : {result}\n")

    except ClientError as e:
        logger.error(f"[!] Endpoint invocation failed: {e}\n")


if __name__ == "__main__":
    logger.info("==================================================================")
    logger.info(f"   SAGEMAKER ENDPOINT DIAGNOSTIC: {SAGEMAKER_ENDPOINT_NAME}")
    logger.info("==================================================================\n")

    for test_case in TEST_CASES:
        test_single_vector(test_case)