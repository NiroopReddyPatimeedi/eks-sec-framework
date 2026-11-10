#!/usr/bin/env python3
"""
AWS SageMaker Isolation Forest Deployment Pipeline
--------------------------------------------------
1. Dynamically resolves AWS Account ID and S3 bucket: eks-ebpf-telemetry-baseline-$ACCOUNT_ID
2. Locates active SageMaker IAM Execution Role (EKS-SageMaker-ExecutionRole).
3. Exports full scaler & decision tree parameters into model.json for version-agnostic execution.
4. Generates code/inference.py with a pure NumPy JSONIsolationForest engine to bypass joblib C-struct mismatches.
5. Packages model.joblib, model.json, inference.py (root + code/), and code/__init__.py into model.tar.gz.
6. Purges failed endpoints, ALL legacy matching endpoint configurations, and obsolete models.
7. Deletes existing endpoint before creation to bypass 25-minute Blue/Green rollback cycles.
8. Uploads model.tar.gz to Amazon S3 in region ap-south-1.
9. Creates fresh SageMaker Endpoint with real-time elapsed wait time logging and 5-min timeout.
"""

import json
import logging
import os
import sys
import tarfile
import time
import boto3
import joblib
import numpy as np
from botocore.exceptions import ClientError
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Optional dynamic container URI lookup via SageMaker SDK
try:
    import sagemaker
    from sagemaker.image_uris import retrieve as retrieve_image_uri
    HAS_SAGEMAKER_SDK = True
except ImportError:
    HAS_SAGEMAKER_SDK = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "eks-isolation-forest-endpoint")
SAGEMAKER_MODEL_NAME = os.environ.get("SAGEMAKER_MODEL_NAME", "eks-isolation-forest-model")
SAGEMAKER_CONFIG_NAME = os.environ.get("SAGEMAKER_CONFIG_NAME", "eks-isolation-forest-config")

MODEL_JOBLIB_PATH = "model.joblib"
MODEL_JSON_PATH = "model.json"
TAR_GZ_PATH = "model.tar.gz"

FEATURE_NAMES = [
    "process_spawn_rate",
    "memory_rss_kb",
    "cpu_pct",
    "syscall_rate",
    "net_io_kb",
    "active_sockets"
]

# Standardized logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SageMaker-Deployer")

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
s3_client = session.client("s3")
sm_client = session.client("sagemaker")
sts_client = session.client("sts")


def resolve_account_and_bucket() -> tuple:
    """Dynamically resolves AWS Account ID and standardizes S3 bucket name."""
    account_id = sts_client.get_caller_identity()["Account"]
    default_bucket_name = f"eks-ebpf-telemetry-baseline-{account_id}"
    bucket_name = os.environ.get("S3_BUCKET_NAME", default_bucket_name)
    return account_id, bucket_name


def resolve_sagemaker_role(account_id: str) -> str:
    """Locates an active SageMaker IAM Execution Role in the AWS account."""
    env_role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if env_role:
        logger.info(f" [*] Using role from SAGEMAKER_ROLE_ARN env: {env_role}")
        return env_role

    iam_client = session.client("iam")
    candidate_role_names = [
        "EKS-SageMaker-ExecutionRole",
        "SageMakerExecutionRole",
        "AmazonSageMaker-ExecutionRole"
    ]

    for role_name in candidate_role_names:
        try:
            res = iam_client.get_role(RoleName=role_name)
            arn = res["Role"]["Arn"]
            logger.info(f" [OK] Discovered active IAM Execution Role: {arn}")
            return arn
        except ClientError:
            continue

    fallback_arn = f"arn:aws:iam::{account_id}:role/EKS-SageMaker-ExecutionRole"
    logger.warning(f"[!] No candidate IAM role matched via API. Defaulting to fallback ARN: {fallback_arn}")
    return fallback_arn


def ensure_inference_code_exists():
    """Generates code/inference.py containing the version-agnostic JSONIsolationForest engine."""
    os.makedirs("code", exist_ok=True)
    
    # Create code/__init__.py package marker
    init_path = os.path.join("code", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# Package marker for SageMaker\n")

    inference_path = os.path.join("code", "inference.py")

    code_content = '''import os
import json
import logging
import sys
import joblib
import numpy as np

# Configure container logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _average_path_length(n_samples):
    """Calculates standard average path length c(n) of BST."""
    n = np.asarray(n_samples, dtype=np.float64)
    avg = np.zeros_like(n)
    m1 = n <= 1
    m2 = n == 2
    m3 = n > 2
    avg[m1] = 0.0
    avg[m2] = 1.0
    avg[m3] = 2.0 * (np.log(n[m3] - 1.0) + 0.5772156649) - (2.0 * (n[m3] - 1.0) / n[m3])
    return avg

class JSONIsolationForest:
    """Pure NumPy IsolationForest + StandardScaler evaluator (version-agnostic)."""
    def __init__(self, model_dict):
        scaler_info = model_dict.get("scaler", {})
        self.mean = np.array(scaler_info["mean"]) if scaler_info.get("mean") else None
        self.scale = np.array(scaler_info["scale"]) if scaler_info.get("scale") else None
        
        iforest_info = model_dict.get("iforest", {})
        self.max_samples = iforest_info.get("max_samples", 100)
        self.offset = iforest_info.get("offset", -0.5)
        self.trees = iforest_info.get("trees", [])
        self.c_max_samples = _average_path_length([self.max_samples])[0]

    def _transform(self, X):
        if self.mean is not None and self.scale is not None:
            return (X - self.mean) / self.scale
        return X

    def decision_function(self, X):
        X_scaled = self._transform(X)
        n_samples = X_scaled.shape[0]
        paths = np.zeros((n_samples, len(self.trees)))

        for i, sample in enumerate(X_scaled):
            for t_idx, tree in enumerate(self.trees):
                node = 0
                depth = 0
                left = tree["left"]
                right = tree["right"]
                feat = tree["feature"]
                thresh = tree["threshold"]
                n_node_samples = tree["n_node_samples"]

                while left[node] != -1 and right[node] != -1:
                    f = feat[node]
                    if sample[f] <= thresh[node]:
                        node = left[node]
                    else:
                        node = right[node]
                    depth += 1

                leaf_samples = n_node_samples[node]
                h = depth + _average_path_length([leaf_samples])[0]
                paths[i, t_idx] = h

        mean_h = np.mean(paths, axis=1)
        score_samples = - (2.0 ** (- mean_h / self.c_max_samples))
        return score_samples - self.offset

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

def model_fn(model_dir):
    """
    Loads model artifact.
    Prioritizes model.json for 100% version-agnostic inference across Scikit-Learn versions.
    Falls back to joblib.load(model.joblib) if JSON is unavailable.
    """
    logger.info(f"[*] Container initializing model load from: {model_dir}")
    
    # 1. Prioritize model.json (Version-Agnostic)
    json_path = os.path.join(model_dir, "model.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(model_dir, "code", "model.json")

    if os.path.exists(json_path):
        try:
            logger.info(f"[*] Loading JSON model representation: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                model_dict = json.load(f)
            
            if "iforest" in model_dict and "trees" in model_dict["iforest"] and len(model_dict["iforest"]["trees"]) > 0:
                model = JSONIsolationForest(model_dict)
                logger.info(f"[OK] Successfully loaded JSON IsolationForest model (Trees: {len(model.trees)})")
                return model
        except Exception as e:
            logger.warning(f"[!] JSON model load failed: {str(e)}. Falling back to joblib...")

    # 2. Fallback to model.joblib
    joblib_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(joblib_path):
        joblib_path = os.path.join(model_dir, "code", "model.joblib")

    if os.path.exists(joblib_path):
        try:
            model = joblib.load(joblib_path)
            logger.info(f"[OK] joblib artifact loaded successfully. Type: {type(model)}")
            return model
        except Exception as e:
            logger.error(f"[!] Exception during joblib.load: {str(e)}", exc_info=True)
            raise e

    raise FileNotFoundError(f"No valid model artifact (model.json or model.joblib) found inside {model_dir}")

def input_fn(request_body, request_content_type):
    """Parses incoming eBPF telemetry JSON payload."""
    logger.info(f"[*] Request content-type: {request_content_type}")
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            data = data["features"]
        
        arr = np.array(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Executes inference through JSONIsolationForest or Pipeline."""
    try:
        raw_pred = model.predict(input_data)[0]
        score = float(model.decision_function(input_data)[0])

        return {
            "is_anomaly": 1 if raw_pred == -1 else 0,
            "raw_predict": int(raw_pred),
            "anomaly_score": round(score, 4)
        }
    except Exception as e:
        logger.error(f"[!] Error during prediction execution: {str(e)}", exc_info=True)
        raise e

def output_fn(prediction, accept):
    """Formats response output into JSON."""
    if accept in ["application/json", "*/*"]:
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
'''
    with open(inference_path, "w", encoding="utf-8") as f:
        f.write(code_content)
    logger.info(f" [OK] Verified inference entrypoint script at '{inference_path}'")


def get_sklearn_image_uri(region: str) -> str:
    """Retrieves official SageMaker Scikit-Learn ECR container image URI."""
    if HAS_SAGEMAKER_SDK:
        try:
            uri = retrieve_image_uri(
                framework="sklearn",
                region=region,
                version="1.2-1",
                py_version="py3",
                instance_type="ml.t2.medium"
            )
            logger.info(f" [*] Dynamic SDK Image URI resolved: {uri}")
            return uri
        except Exception as e:
            logger.warning(f"[!] SDK image lookup failed ({e}). Falling back to region mapping...")

    # Official AWS Registry IDs for sagemaker-scikit-learn:1.2-1-cpu-py3
    sklearn_account_map = {
        "ap-south-1": "720646828776",      # Asia Pacific (Mumbai)
        "us-east-1": "683313688378",       # US East (N. Virginia)
        "us-east-2": "257758044811",       # US East (Ohio)
        "us-west-2": "246696594541",       # US West (Oregon)
        "eu-west-1": "141502664222",       # Europe (Ireland)
    }
    registry_account = sklearn_account_map.get(region, "720646828776")
    uri = f"{registry_account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    logger.info(f" [*] Regional Mapped Image URI: {uri}")
    return uri


def convert_and_package_artifacts():
    """Inspects model.joblib, extracts complete tree data into model.json, and packages model.tar.gz."""
    if not os.path.exists(MODEL_JOBLIB_PATH):
        raise FileNotFoundError(f"Missing required artifact: '{MODEL_JOBLIB_PATH}'")

    ensure_inference_code_exists()

    logger.info(f"[*] Inspecting artifact: '{MODEL_JOBLIB_PATH}'...")
    loaded_artifact = joblib.load(MODEL_JOBLIB_PATH)

    if isinstance(loaded_artifact, Pipeline) or hasattr(loaded_artifact, "named_steps"):
        logger.info(" [OK] Identified Scikit-Learn Pipeline object (Scaler + Model).")
        iforest = loaded_artifact.named_steps.get("model", loaded_artifact.steps[-1][1])
        scaler = loaded_artifact.named_steps.get("scaler", None)
    elif isinstance(loaded_artifact, IsolationForest):
        logger.info(" [*] Identified standalone IsolationForest object.")
        iforest = loaded_artifact
        scaler = None
    else:
        raise TypeError(f"Unrecognized artifact object type: {type(loaded_artifact)}")

    # Extract scaler parameters
    scaler_dict = {}
    if scaler is not None and hasattr(scaler, "mean_"):
        scaler_dict = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist()
        }

    # Extract full decision tree parameters for version-agnostic JSON evaluation
    trees_data = []
    if hasattr(iforest, "estimators_"):
        for est in iforest.estimators_:
            t = est.tree_
            trees_data.append({
                "left": t.children_left.tolist(),
                "right": t.children_right.tolist(),
                "feature": t.feature.tolist(),
                "threshold": t.threshold.tolist(),
                "n_node_samples": t.n_node_samples.tolist()
            })

    model_metadata = {
        "version": "1.0",
        "model_type": "IsolationForest",
        "is_pipeline": isinstance(loaded_artifact, Pipeline),
        "feature_names": FEATURE_NAMES,
        "scaler": scaler_dict,
        "iforest": {
            "max_samples": int(getattr(iforest, "max_samples_", 100)),
            "offset": float(getattr(iforest, "offset_", -0.5)),
            "contamination": float(getattr(iforest, "contamination", 0.05)),
            "n_estimators": len(trees_data),
            "trees": trees_data
        }
    }

    with open(MODEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f" [OK] Exported version-agnostic JSON model: '{MODEL_JSON_PATH}' (Trees: {len(trees_data)})")

    logger.info(f" [*] Packaging '{TAR_GZ_PATH}' containing model.joblib, model.json, inference.py...")
    with tarfile.open(TAR_GZ_PATH, "w:gz") as tar:
        # Package model artifacts
        tar.add(MODEL_JOBLIB_PATH, arcname="model.joblib")
        tar.add(MODEL_JSON_PATH, arcname="model.json")
        
        # Package inference script at both root AND code/ directory for compatibility
        tar.add("code/inference.py", arcname="inference.py")
        tar.add("code/inference.py", arcname="code/inference.py")
        if os.path.exists("code/__init__.py"):
            tar.add("code/__init__.py", arcname="code/__init__.py")

    logger.info(f" [OK] Package creation complete: '{TAR_GZ_PATH}'")


def cleanup_stale_sagemaker_resources():
    """
    Purges failed endpoints, ALL existing matching endpoint configurations,
    and legacy models before building new deployment resources.
    """
    logger.info(" [*] Purging stale SageMaker resources...")

    # 1. Delete Endpoint if failed or rolled back
    try:
        resp = sm_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        if status in ["Failed", "RolledBack"]:
            logger.warning(f"[!] Endpoint '{SAGEMAKER_ENDPOINT_NAME}' is in '{status}' state. Deleting...")
            sm_client.delete_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
            time.sleep(5)
    except ClientError:
        pass

    # 2. Delete ALL existing endpoint configurations matching prefix
    try:
        paginator = sm_client.get_paginator("list_endpoint_configs")
        deleted_configs = 0
        for page in paginator.paginate(NameContains=SAGEMAKER_CONFIG_NAME):
            for cfg in page["EndpointConfigs"]:
                cfg_name = cfg["EndpointConfigName"]
                try:
                    sm_client.delete_endpoint_config(EndpointConfigName=cfg_name)
                    logger.info(f" [-] Deleted EndpointConfig: '{cfg_name}'")
                    deleted_configs += 1
                except ClientError:
                    logger.info(f" [i] Kept active EndpointConfig in use: '{cfg_name}'")
        logger.info(f" [OK] EndpointConfig cleanup complete (Deleted: {deleted_configs})")
    except Exception as e:
        logger.warning(f"[!] Cleanup notice (EndpointConfigs): {e}")

    # 3. Purge older SageMaker models matching model prefix
    try:
        paginator = sm_client.get_paginator("list_models")
        deleted_models = 0
        for page in paginator.paginate(NameContains=SAGEMAKER_MODEL_NAME):
            for m in page["Models"]:
                name = m["ModelName"]
                try:
                    sm_client.delete_model(ModelName=name)
                    logger.info(f" [-] Deleted legacy Model: '{name}'")
                    deleted_models += 1
                except ClientError:
                    pass
        logger.info(f" [OK] Model cleanup complete (Deleted: {deleted_models})")
    except Exception as e:
        logger.warning(f"[!] Cleanup notice (Models): {e}")


def upload_artifact_to_s3(bucket_name: str) -> str:
    """Ensures S3 bucket exists and uploads model.tar.gz."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        logger.info(f" [*] Bucket '{bucket_name}' not found. Creating in region '{AWS_REGION}'...")
        if AWS_REGION == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )

    s3_key = f"models/{SAGEMAKER_MODEL_NAME}/{TAR_GZ_PATH}"
    logger.info(f" [*] Uploading '{TAR_GZ_PATH}' -> s3://{bucket_name}/{s3_key}...")
    s3_client.upload_file(TAR_GZ_PATH, bucket_name, s3_key)
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    logger.info(f" [OK] Upload successful: {s3_uri}")
    return s3_uri


def deploy_sagemaker_endpoint(s3_model_uri: str, role_arn: str, timeout_seconds: int = 300):
    """
    Deletes existing endpoint (bypassing 25-minute Blue/Green rollbacks) and creates
    a clean fresh SageMaker Endpoint with a 5-minute timeout.
    """
    image_uri = get_sklearn_image_uri(AWS_REGION)

    timestamp = int(time.time())
    model_name = f"{SAGEMAKER_MODEL_NAME}-{timestamp}"
    
    logger.info(f" [*] Registering SageMaker Model: '{model_name}'...")
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": s3_model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": s3_model_uri
            }
        }
    )

    config_name = f"{SAGEMAKER_CONFIG_NAME}-{timestamp}"
    logger.info(f" [*] Creating Endpoint Config: '{config_name}'...")
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t2.medium",
                "InitialVariantWeight": 1.0
            }
        ]
    )

    # Delete existing endpoint prior to creation to prevent Blue/Green rollback overhead
    try:
        resp = sm_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        curr_status = resp["EndpointStatus"]
        logger.info(f" [*] Existing endpoint '{SAGEMAKER_ENDPOINT_NAME}' found (Status: {curr_status}).")
        logger.info(" [*] Issuing deletion request to enforce clean creation (bypassing 25m Blue/Green rollback)...")
        sm_client.delete_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        
        logger.info(" [*] Waiting for endpoint deletion to complete...")
        while True:
            try:
                sm_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
                time.sleep(5)
            except ClientError:
                logger.info(" [OK] Previous endpoint successfully deleted.")
                break
    except ClientError:
        pass

    logger.info(f" [*] Creating fresh SageMaker Endpoint '{SAGEMAKER_ENDPOINT_NAME}' (Timeout: {timeout_seconds // 60}m)...")
    sm_client.create_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        EndpointConfigName=config_name
    )

    start_time = time.time()

    while True:
        status_resp = sm_client.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        status = status_resp["EndpointStatus"]
        elapsed_seconds = int(time.time() - start_time)
        mins, secs = divmod(elapsed_seconds, 60)
        
        logger.info(f"     Status: {status} | Elapsed Time: {mins:02d}m {secs:02d}s...")
        
        if status == "InService":
            logger.info(f"\n[OK] SageMaker Endpoint '{SAGEMAKER_ENDPOINT_NAME}' is live and InService! Total Time: {mins:02d}m {secs:02d}s")
            break
        elif status in ["Failed", "RolledBack"]:
            failure_reason = status_resp.get("FailureReason", "Unknown error")
            raise RuntimeError(f"Endpoint deployment failed ({status}): {failure_reason}")
            
        if elapsed_seconds >= timeout_seconds:
            logger.error(f"\n[!] Deployment timeout threshold ({timeout_seconds // 60} mins) reached. Endpoint status: '{status}'.")
            raise TimeoutError(f"SageMaker endpoint deployment timed out after {mins:02d}m {secs:02d}s (Status: '{status}').")

        time.sleep(15)


def main():
    logger.info("==================================================================")
    logger.info("      AWS SAGEMAKER ISOLATION FOREST DEPLOYMENT PIPELINE          ")
    logger.info("==================================================================")

    account_id, bucket_name = resolve_account_and_bucket()
    role_arn = resolve_sagemaker_role(account_id)

    logger.info(f" [*] AWS Account ID : {account_id}")
    logger.info(f" [*] S3 Bucket Name : {bucket_name}")
    logger.info(f" [*] Execution Role : {role_arn}")
    logger.info(f" [*] Target Region  : {AWS_REGION}")
    logger.info(f" [*] Target Endpoint: {SAGEMAKER_ENDPOINT_NAME}\n")

    cleanup_stale_sagemaker_resources()
    convert_and_package_artifacts()
    s3_uri = upload_artifact_to_s3(bucket_name)
    deploy_sagemaker_endpoint(s3_uri, role_arn, timeout_seconds=300)


if __name__ == "__main__":
    main()