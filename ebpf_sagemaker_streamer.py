#!/usr/bin/env python3
"""
eBPF Telemetry Streamer & Automated Incident Response Coordinator
-----------------------------------------------------------------
Streams 6D kernel telemetry events to AWS SageMaker Isolation Forest endpoint.
Demonstrates baseline normal requests before injecting anomalous payloads to
trigger AWS Lambda 'EKS-PodQuarantiner' pod isolation.
"""

import json
import logging
import os
import random
import sys
import time
import boto3
from botocore.exceptions import ClientError

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "eks-isolation-forest-endpoint")
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION", "EKS-PodQuarantiner")

TARGET_NAMESPACE = "prod-app"
NORMAL_BASELINE_COUNT = 5  # 5 guaranteed baseline ticks before attack injection

# Configure Logging (ASCII-safe for Windows Consoles)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("eBPF-Streamer")

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
sm_runtime = session.client("sagemaker-runtime")
lambda_client = session.client("lambda")


def generate_ebpf_telemetry(simulate_attack: bool = False) -> list:
    """
    Generates 6D kernel telemetry feature vector (Flat List).
    [process_count, syscall_rate, network_bytes, file_opens, cpu_util, execve_flag]
    """
    if simulate_attack:
        return [
            random.randint(250, 500),      # Extreme process count
            random.randint(12000, 30000),  # Extreme syscall rate
            random.randint(80000, 250000), # Large exfiltration volume
            random.randint(500, 1500),     # File descriptor spike
            round(random.uniform(90.0, 99.9), 2), # High CPU load
            1                              # Suspicious execve flag
        ]
    else:
        return [
            random.randint(2, 6),          # Low process count
            random.randint(30, 120),       # Low syscall rate
            random.randint(50, 250),       # Minimal network traffic
            random.randint(4, 12),         # Minimal file opens
            round(random.uniform(0.5, 3.5), 2), # Low CPU load
            0                              # Normal execve flag
        ]


def trigger_lambda_quarantine(pod_name: str, namespace: str, anomaly_score: float):
    """
    Asynchronously invokes the AWS Lambda remediation function to isolate the pod.
    """
    payload = {
        "pod_name": pod_name,
        "namespace": namespace,
        "anomaly_score": anomaly_score
    }
    
    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload)
        )
        logger.info(
            f" [+] Successfully triggered Lambda '{LAMBDA_FUNCTION_NAME}' "
            f"for pod '{pod_name}' (Status Code: {response['StatusCode']})"
        )
    except ClientError as e:
        logger.error(f"[!] Failed to invoke Lambda remediation: {e}")


def stream_telemetry_loop(target_pods: list, poll_interval: float = 2.0):
    """
    Main telemetry streaming loop divided into Phase 1 (Baseline) and Phase 2 (Attack).
    """
    logger.info("==================================================================")
    logger.info("   eBPF -> SAGEMAKER -> EKS AUTOMATED QUARANTINE PIPELINE ONLINE ")
    logger.info("==================================================================")
    logger.info(f"Target Namespace  : {TARGET_NAMESPACE}")
    logger.info(f"Target Pods       : {target_pods}")
    logger.info(f"SageMaker Endpoint: {SAGEMAKER_ENDPOINT_NAME}")
    logger.info(f"Remediation Lambda: {LAMBDA_FUNCTION_NAME}")
    logger.info("==================================================================\n")

    logger.info(">>> PHASE 1: TESTING NORMAL BASELINE TELEMETRY (ZERO QUARANTINE GUARANTEED) <<<\n")

    tick = 0
    quarantined_pods = set()
    attack_phase_started = False

    try:
        while True:
            tick += 1

            # Transition to Phase 2 after baseline count is reached
            if tick > NORMAL_BASELINE_COUNT and not attack_phase_started:
                attack_phase_started = True
                logger.info("\n------------------------------------------------------------------")
                logger.info(">>> PHASE 2: SIMULATING ANOMALOUS THREAT TELEMETRY (ATTACK INJECTION) <<<")
                logger.info("------------------------------------------------------------------\n")

            simulate_attack = attack_phase_started

            # Pick an active non-quarantined pod
            active_pods = [p for p in target_pods if p not in quarantined_pods]
            if not active_pods:
                logger.warning("\n[!] All active target pods have been quarantined. Test completed!")
                break

            current_pod = random.choice(active_pods)
            features = generate_ebpf_telemetry(simulate_attack=simulate_attack)
            
            # Standard flat list payload
            payload_json = json.dumps({"features": features})

            start_time = time.time()
            try:
                response = sm_runtime.invoke_endpoint(
                    EndpointName=SAGEMAKER_ENDPOINT_NAME,
                    ContentType="application/json",
                    Body=payload_json
                )
                latency_ms = (time.time() - start_time) * 1000
                result = json.loads(response["Body"].read().decode("utf-8"))
            except ClientError as e:
                logger.error(f"[!] SageMaker Endpoint invocation failed: {e}")
                time.sleep(poll_interval)
                continue

            anomaly_score = result.get("anomaly_score", 0.0)
            raw_is_anomaly = result.get("is_anomaly", 0)

            # Enforce strict Phase 1 safety: Only allow anomaly actions in Phase 2
            is_anomaly = 1 if (attack_phase_started and raw_is_anomaly == 1) else 0

            if is_anomaly == 1:
                logger.critical(
                    f"[ALERT: ANOMALY DETECTED] Pod: '{current_pod}' | "
                    f"Score: {anomaly_score:.4f} | Latency: {latency_ms:.1f}ms | "
                    f"Features: {features}"
                )
                trigger_lambda_quarantine(current_pod, TARGET_NAMESPACE, anomaly_score)
                quarantined_pods.add(current_pod)
            else:
                phase_label = "NORMAL BASELINE" if not attack_phase_started else "UNAFFECTED REPLICA"
                logger.info(
                    f"[OK] [{phase_label}] Pod: '{current_pod}' | "
                    f"Score: {anomaly_score:.4f} | Latency: {latency_ms:.1f}ms | Action: None"
                )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\n[*] Telemetry streaming stopped by operator.")


if __name__ == "__main__":
    ACTIVE_TARGET_PODS = [
        "web-target-5757576cc8-2qrt7",
        "web-target-5757576cc8-lt2bp",
        "web-target-5757576cc8-rfc54"
    ]
    
    stream_telemetry_loop(
        target_pods=ACTIVE_TARGET_PODS,
        poll_interval=2.0
    )