import os
import json
import base64
import boto3
import urllib3
from botocore.signers import RequestSigner

http = urllib3.PoolManager()

# Environment Variables
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "eks-insider-threat-detector")
EKS_CLUSTER_NAME = os.getenv("EKS_CLUSTER_NAME", "eks-security-cluster")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
eks_client = boto3.client("eks", region_name=AWS_REGION)

def get_eks_bearer_token(cluster_name):
    """Generates an IAM authentication bearer token for Kubernetes API request signing."""
    session = boto3.session.Session()
    sts_client = session.client("sts", region_name=AWS_REGION)
    signer = RequestSigner(
        sts_client._client_config.service_name,
        AWS_REGION,
        "sts",
        "v4",
        sts_client._client_config.credentials,
        sts_client._client_config.request_signer._signer.driver
    )
    
    params = {
        "method": "GET",
        "url": f"https://sts.{AWS_REGION}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
        "body": {},
        "headers": {"x-k8s-aws-id": cluster_name},
        "context": {}
    }
    
    signed_url = signer.generate_presigned_url(params, region_name=AWS_REGION, expires_in=60, operation_name="")
    base64_url = base64.urlsafe_b64encode(signed_url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"k8s-aws-v1.{base64_url}"

def patch_pod_label(api_server, token, namespace, pod_name):
    """Applies isolation label security.status=quarantined to target pod."""
    url = f"{api_server}/api/v1/namespaces/{namespace}/pods/{pod_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/strategic-merge-patch+json",
        "Accept": "application/json"
    }
    patch_body = json.dumps({
        "metadata": {
            "labels": {
                "security.status": "quarantined"
            }
        }
    })
    
    res = http.request("PATCH", url, headers=headers, body=patch_body, verify=False)
    print(f" [*] Pod Label Patch Response: {res.status}")

def apply_quarantine_network_policy(api_server, token, namespace):
    """Ensures restrictive NetworkPolicy blocking all Ingress and Egress on quarantined pods exists."""
    url = f"{api_server}/apis/networking.k8s.io/v1/namespaces/{namespace}/networkpolicies"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    policy_manifest = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": "isolate-quarantined-pods",
            "namespace": namespace
        },
        "spec": {
            "podSelector": {
                "matchLabels": {
                    "security.status": "quarantined"
                }
            },
            "policyTypes": ["Ingress", "Egress"]
        }
    }
    
    res = http.request("POST", url, headers=headers, body=json.dumps(policy_manifest), verify=False)
    if res.status in [200, 201]:
        print(f" [✓] Created Quarantine NetworkPolicy in namespace '{namespace}'")
    elif res.status == 409:
        print(f" [✓] Quarantine NetworkPolicy already active in namespace '{namespace}'")

def lambda_handler(event, context):
    """Main EventBridge Trigger Handler."""
    print("==========================================================================")
    print("EVENTBRIDGE THREAT DETECTION & EKS ISOLATION ENGINE")
    print("==========================================================================")
    
    # Extract telemetry window details from EventBridge Payload
    detail = event.get("detail", {})
    pod_name = detail.get("pod_name")
    namespace = detail.get("namespace", "default")
    feature_vector = detail.get("features") # List of 6 values
    
    if not pod_name or not feature_vector:
        print(" [!] Missing mandatory fields (pod_name or features). Exiting.")
        return {"status": "SKIPPED_INVALID_PAYLOAD"}
    
    # 1. Query SageMaker Real-Time Endpoint
    payload = json.dumps({"features": feature_vector})
    sm_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=payload
    )
    
    result = json.loads(sm_response["Body"].read().decode("utf-8"))
    is_anomaly = result.get("is_anomaly", 0)
    anomaly_score = result.get("anomaly_score", 0.0)
    
    print(f" [*] Target Pod        : {pod_name} ({namespace})")
    print(f" [*] Prediction Output : Anomaly={is_anomaly} | Score={anomaly_score:.4f}")
    
    # 2. Trigger Isolation Sequence if Threat Detected
    if is_anomaly == 1:
        print(f" [🚨] CRITICAL THREAT DETECTED ON POD: {pod_name}. INITIATING ISOLATION...")
        
        # Resolve EKS Endpoint
        cluster_info = eks_client.describe_cluster(name=EKS_CLUSTER_NAME)["cluster"]
        api_server = cluster_info["endpoint"]
        
        # Get IAM Bearer Token
        token = get_eks_bearer_token(EKS_CLUSTER_NAME)
        
        # Apply Network Policy & Label Pod
        apply_quarantine_network_policy(api_server, token, namespace)
        patch_pod_label(api_server, token, namespace, pod_name)
        
        print(f" [✓] ISOLATION COMPLETE: Pod '{pod_name}' trapped under restrictive NetworkPolicy.")
        return {"status": "POD_QUARANTINED", "pod": pod_name, "score": anomaly_score}
    
    print(" [✓] Telemetry evaluated as Normal. No isolation action required.")
    return {"status": "NORMAL", "score": anomaly_score}