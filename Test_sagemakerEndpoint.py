import json
import boto3

sm_runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')

# Keys can be sent in ANY order
telemetry_payload = {
    "features": {
        "cpu_pct": 45.2,
        "syscall_rate": 8.5,
        "process_spawn_rate": 15.0,
        "active_sockets": 120.0,
        "memory_rss_kb": 450000.0,
        "net_io_kb": 3.2
    }
}

response = sm_runtime.invoke_endpoint(
    EndpointName="eks-isolation-forest-endpoint",
    ContentType="application/json",
    Body=json.dumps(telemetry_payload)
)

result = json.loads(response['Body'].read().decode('utf-8'))
print(json.dumps(result, indent=2))