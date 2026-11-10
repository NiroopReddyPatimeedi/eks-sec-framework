import os
import json
import numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at {model_path}")
    with open(model_path, 'r') as f:
        return json.load(f)

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            return data["features"]
        return data
    raise ValueError(f"Unsupported content_type: {request_content_type}")

def c_calc(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

def predict_fn(input_data, model_dict):
    c_N = model_dict['c_N']
    offset_ = model_dict['offset_']
    trees = model_dict['trees']
    scaler = model_dict.get('scaler')
    feature_names = model_dict.get('feature_names', [])

    # Format input data into 2D float64 NumPy array
    if isinstance(input_data, dict):
        if not feature_names:
            raise KeyError("Dictionary input provided, but 'feature_names' is missing from model.json")
        ordered_vec = [float(input_data[col]) for col in feature_names]
        input_data = np.array([ordered_vec], dtype=np.float64)
    elif isinstance(input_data, list):
        if len(input_data) > 0 and isinstance(input_data[0], dict):
            if not feature_names:
                raise KeyError("Dictionary batch input provided, but 'feature_names' is missing from model.json")
            ordered_batch = [[float(item[col]) for col in feature_names] for item in input_data]
            input_data = np.array(ordered_batch, dtype=np.float64)
        else:
            arr = np.array(input_data, dtype=np.float64)
            input_data = arr if arr.ndim == 2 else np.expand_dims(arr, axis=0)
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            input_data = np.expand_dims(input_data, axis=0)
    else:
        raise ValueError(f"Unsupported input data structure: {type(input_data)}")

    # Apply scaling if scaler is present: (x - mean) / scale
    if scaler is not None:
        mean = np.array(scaler['mean'], dtype=np.float64)
        scale = np.array(scaler['scale'], dtype=np.float64)
        input_data = (input_data - mean) / scale

    results = []
    for x in input_data:
        paths = []
        for t in trees:
            node = 0
            depth = 0
            c_left = t['children_left']
            c_right = t['children_right']
            features = t['feature']
            thresholds = t['threshold']
            n_samples = t['n_node_samples']

            while features[node] >= 0:
                if x[features[node]] <= thresholds[node]:
                    node = c_left[node]
                else:
                    node = c_right[node]
                depth += 1

            leaf_n = n_samples[node]
            c_leaf = c_calc(leaf_n)
            paths.append(depth + c_leaf)

        mean_path = float(np.mean(paths))
        raw_score = float(2 ** (- mean_path / c_N))
        decision_score = float(- raw_score - offset_)
        is_anomaly = 1 if decision_score < 0.0 else 0

        results.append({
            "is_anomaly": is_anomaly,
            "anomaly_score": float(round(decision_score, 4))
        })
    return results[0] if len(results) == 1 else results

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
