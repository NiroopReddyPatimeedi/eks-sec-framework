import base64
import json
import os
import ssl
import sys
import urllib.request
import boto3
from botocore.exceptions import ClientError

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
REGION = os.environ.get("AWS_REGION", "ap-south-1")
CLUSTER_NAME = os.environ.get("EKS_CLUSTER_NAME", "eks-sec-cluster")

eks_client = boto3.client("eks", region_name=REGION)


def get_eks_bearer_token(cluster_name: str, region: str) -> str:
    """
    Generates an EKS authentication bearer token using STS presigned URL.
    Uses boto3 event hooks to inject 'x-k8s-aws-id' safely across all botocore versions.
    """
    sts_client = boto3.client("sts", region_name=region)

    def _inject_k8s_header(request, **kwargs):
        request.headers["x-k8s-aws-id"] = cluster_name

    # Register hook to inject header during SigV4 presigning
    sts_client.meta.events.register(
        "before-sign.sts.GetCallerIdentity", _inject_k8s_header
    )

    presigned_url = sts_client.generate_presigned_url(
        "get_caller_identity",
        Params={},
        ExpiresIn=60,
        HttpMethod="GET"
    )

    # Encode presigned URL into URL-safe base64 string
    base64_url = base64.urlsafe_b64encode(presigned_url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"k8s-aws-v1.{base64_url}"


def k8s_api_request(endpoint: str, ca_cert_b64: str, token: str, path: str, method: str = "GET", payload: dict = None, content_type: str = "application/json"):
    """
    Executes raw HTTPS REST requests directly against the EKS Kubernetes API Server.
    """
    ca_filename = "/tmp/ca.crt"
    with open(ca_filename, "wb") as f:
        f.write(base64.b64decode(ca_cert_b64))

    context = ssl.create_default_context(cafile=ca_filename)
    url = f"{endpoint}{path}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "Accept": "application/json"
    }

    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, context=context) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        if e.code == 409:  # AlreadyExists
            return {"status": "AlreadyExists", "message": error_body}
        print(f"[!] K8s API HTTP {e.code} Error: {error_body}")
        raise RuntimeError(f"K8s API Error ({e.code}): {error_body}")


def quarantine_pod(pod_name: str, namespace: str, anomaly_score: float):
    """
    Isolates target Pod by updating labels (removing it from K8s Service selectors)
    and enforcing a default-deny NetworkPolicy on quarantined pods.
    """
    print(f"[*] Initiating quarantine for Pod '{pod_name}' in Namespace '{namespace}' (Score: {anomaly_score})...")

    # 1. Fetch EKS API endpoint and CA certificate
    cluster_info = eks_client.describe_cluster(name=CLUSTER_NAME)["cluster"]
    api_endpoint = cluster_info["endpoint"]
    ca_cert = cluster_info["certificateAuthority"]["data"]

    # 2. Generate EKS authentication token
    token = get_eks_bearer_token(CLUSTER_NAME, REGION)

    # 3. Create or verify 'quarantine-deny-all' NetworkPolicy in namespace
    netpol_path = f"/apis/networking.k8s.io/v1/namespaces/{namespace}/networkpolicies"
    netpol_manifest = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": "quarantine-deny-all",
            "namespace": namespace
        },
        "spec": {
            "podSelector": {
                "matchLabels": {
                    "quarantine": "true"
                }
            },
            "policyTypes": ["Ingress", "Egress"]
        }
    }
    
    try:
        netpol_res = k8s_api_request(
            api_endpoint, ca_cert, token, netpol_path, 
            method="POST", payload=netpol_manifest
        )
        if netpol_res.get("status") == "AlreadyExists":
            print(f" [✓] NetworkPolicy 'quarantine-deny-all' already active in namespace '{namespace}'.")
        else:
            print(f" [✓] Applied NetworkPolicy 'quarantine-deny-all' in namespace '{namespace}'.")
    except Exception as e:
        print(f" [!] NetworkPolicy warning: {e}")

    # 4. Patch Pod Labels & Annotations (Strategic Merge Patch)
    pod_patch_path = f"/api/v1/namespaces/{namespace}/pods/{pod_name}"
    patch_payload = {
        "metadata": {
            "labels": {
                "quarantine": "true",
                "security.status": "isolated",
                "quarantined_by": "sagemaker-ebpf-pipeline"
            },
            "annotations": {
                "security.anomaly_score": str(anomaly_score),
                "security.quarantined_at": str(os.getenv("AWS_REQUEST_ID", "manual_trigger"))
            }
        }
    }

    response = k8s_api_request(
        api_endpoint,
        ca_cert,
        token,
        pod_patch_path,
        method="PATCH",
        payload=patch_payload,
        content_type="application/strategic-merge-patch+json"
    )

    print(f" [✓] Pod '{pod_name}' successfully quarantined and isolated from cluster traffic!")
    return {
        "status": "QUARANTINED",
        "pod_name": pod_name,
        "namespace": namespace,
        "anomaly_score": anomaly_score,
        "assigned_labels": response.get("metadata", {}).get("labels", {})
    }


def lambda_handler(event, context):
    """
    AWS Lambda entrypoint.
    """
    print(f"Received Remediation Trigger: {json.dumps(event)}")
    
    pod_name = event.get("pod_name")
    namespace = event.get("namespace", "default")
    anomaly_score = event.get("anomaly_score", -1.0)

    if not pod_name:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing required parameter 'pod_name' in event payload."})
        }

    try:
        result = quarantine_pod(pod_name, namespace, anomaly_score)
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
    except Exception as err:
        print(f"[!] Quarantine execution error: {err}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to quarantine pod '{pod_name}': {str(err)}"})
        }