import os
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from train_deep_model import MQTTNet

def predict_sample(model_path, data_file=None, num_samples=5, model_type="auto"):
    """
    Run prediction on MQTT flows using a trained model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    is_pytorch = model_path.endswith('.pt') or model_type == "deep"
    
    if is_pytorch:
        meta_path = model_path.replace('.pt', '_meta.joblib')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta = joblib.load(meta_path)
        scaler = meta['scaler']
        feature_names = meta['feature_names']
        class_names = meta['class_names']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MQTTNet(input_dim=meta['input_dim'], num_classes=meta['num_classes']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        artifact = joblib.load(model_path)
        model = artifact['model']
        scaler = artifact['scaler']
        feature_names = artifact['feature_names']
        class_names = artifact['class_names']
        
    print(f"[*] Loaded model: {model_path}")
    print(f"[*] Expected features ({len(feature_names)}): {feature_names[:5]} ...")
    print(f"[*] Classes: {class_names}\n")
    
    if data_file and os.path.exists(data_file):
        print(f"[*] Reading test flow data from: {data_file}")
        df = pd.read_csv(data_file)
        if len(df) > num_samples:
            df = df.head(num_samples)
    else:
        print(f"[*] No custom data file provided. Extracting sample test rows (Normal and each attack type)...")
        test_samples = []
        for file_key in ['normal', 'scan_A', 'scan_sU', 'sparta', 'mqtt_bruteforce']:
            fname = f"biflow_{file_key}.csv"
            candidates = [
                os.path.join("data", "biflow", fname),
                os.path.join("data", fname),
                fname
            ]
            f_path = next((p for p in candidates if os.path.exists(p)), None)
            if f_path:
                temp_df = pd.read_csv(f_path)
                if file_key == 'normal':
                    sample_row = temp_df[temp_df['is_attack'] == 0].iloc[10:11].copy()
                else:
                    sample_row = temp_df[temp_df['is_attack'] == 1].iloc[10:11].copy()
                sample_row['source_file'] = f"{f_path} ({file_key})"
                test_samples.append(sample_row)
        df = pd.concat(test_samples, ignore_index=True)
        
    # Align features
    available_cols = [c for c in feature_names if c in df.columns]
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        print(f"[!] Warning: Missing features: {missing_cols}. Filling with 0.0.")
        for col in missing_cols:
            df[col] = 0.0
            
    X_input_df = df[feature_names]
    X_scaled = scaler.transform(X_input_df)
    
    if is_pytorch:
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X_scaled).to(device)
            logits = model(tensor_X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
    else:
        pred_indices = model.predict(X_scaled)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)
        else:
            probs = None
            
    print(f"{'#':<3} | {'Source / Context':<28} | {'Predicted Attack Type':<18} | {'Confidence':<10}")
    print("-" * 68)
    for i in range(len(df)):
        pred_class = class_names[pred_indices[i]]
        conf_str = f"{probs[i][pred_indices[i]] * 100:.2f}%" if probs is not None else "N/A"
        context = df['source_file'].iloc[i] if 'source_file' in df.columns else f"Row {i+1}"
        print(f"{i+1:<3} | {context:<28} | {pred_class:<18} | {conf_str:<10}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference / Prediction for MQTT Intrusion Detection")
    parser.add_argument('--model_path', type=str, default='models/best_mqtt_model_biflow_multiclass.joblib')
    parser.add_argument('--data_file', type=str, default=None, help='Path to CSV file with MQTT flow features')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='auto', choices=['auto', 'ml', 'deep'])
    args = parser.parse_args()
    
    predict_sample(args.model_path, args.data_file, args.num_samples, args.model_type)
