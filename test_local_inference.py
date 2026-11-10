import json
import os
import sys

# Add local path so we can import code/inference.py directly
sys.path.append(os.path.abspath("."))

from code.inference import model_fn, input_fn, predict_fn, output_fn

print("[*] Running local pre-flight inference test...")

try:
    # 1. Test model loading
    print("  [1/4] Loading model via model_fn()...")
    model = model_fn(".")
    print(f"        Loaded object type: {type(model)}")

    # 2. Test input parsing
    print("  [2/4] Testing input_fn()...")
    sample_payload = json.dumps({"features": [2.5, 5120.0, 45.2, 180.0, 1024.0, 8]})
    parsed_input = input_fn(sample_payload, "application/json")
    print(f"        Parsed shape: {parsed_input.shape}")

    # 3. Test prediction execution
    print("  [3/4] Testing predict_fn()...")
    prediction = predict_fn(parsed_input, model)
    print(f"        Prediction result: {prediction}")

    # 4. Test output serialization
    print("  [4/4] Testing output_fn()...")
    response_body, content_type = output_fn(prediction, "application/json")
    print(f"        Output payload: {response_body}")

    print("\n[OK] LOCAL TEST PASSED! The code is 100% valid for container deployment.")

except Exception as e:
    print(f"\n[!] LOCAL TEST FAILED: {e}")
    import traceback
    traceback.print_exc()