import os
import sys
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Normal', 'Scan_A', 'Scan_sU', 'Sparta', 'MQTT_Bruteforce']

ATTACK_CONFIG = [
    ('normal', 0, 0),
    ('scan_A', 1, 1),
    ('scan_sU', 1, 2),
    ('sparta', 1, 3),
    ('mqtt_bruteforce', 1, 4)
]

def generate_split_csvs(output_dir="data/splits", val_size=0.15, test_size=0.15, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Reading biflow dataset from data/biflow/ ...")
    
    dfs = []
    for file_key, attack_filter, multi_label in ATTACK_CONFIG:
        fname = f"biflow_{file_key}.csv"
        fpath = os.path.join("data", "biflow", fname)
        if not os.path.exists(fpath):
            fpath = os.path.join("data", fname)
        if not os.path.exists(fpath):
            fpath = fname
            
        print(f"    - Reading {fpath}...")
        df = pd.read_csv(fpath)
        df = df[df['is_attack'] == attack_filter].copy()
        df['target'] = multi_label
        df['class_name'] = CLASS_NAMES[multi_label]
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    total_samples = len(full_df)
    print(f"[*] Total dataset size: {total_samples:,} samples.")
    
    # Drop identifier columns (avoid bias)
    cols_to_drop = ['ip_src', 'ip_dst', 'src_ip', 'dst_ip', 'proto', 'protocol', 'timestamp']
    for c in cols_to_drop:
        if c in full_df.columns:
            full_df.drop(columns=[c], inplace=True)
            
    # Handle NaN / Inf
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.fillna(-1.0)
    
    # 2-step Stratified Split
    print(f"[*] Splitting dataset: Train ({1.0 - val_size - test_size:.0%}) | Val ({val_size:.0%}) | Test ({test_size:.0%})...")
    
    # Step 1: Split out Test set (15%)
    temp_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        random_state=seed,
        stratify=full_df['target']
    )
    
    # Step 2: Split remaining into Train (70%) and Val (15%)
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        temp_df,
        test_size=val_relative_size,
        random_state=seed,
        stratify=temp_df['target']
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    print(f"[*] Saving standalone files:")
    print(f"    [+] Writing {train_path} ({len(train_df):,} samples)...")
    train_df.to_csv(train_path, index=False)
    
    print(f"    [+] Writing {val_path} ({len(val_df):,} samples)...")
    val_df.to_csv(val_path, index=False)
    
    print(f"    [+] Writing {test_path} ({len(test_df):,} samples)...")
    test_df.to_csv(test_path, index=False)
    
    print("\n" + "=" * 80)
    print("                     XUAT CAC FILE DU LIEU HOAN TAT")
    print("=" * 80)
    print(f" 1. Tập TRAIN : {train_path}  ({len(train_df):>7,} dòng, {os.path.getsize(train_path) / 1024**2:.1f} MB)")
    print(f" 2. Tập VAL   : {val_path}    ({len(val_df):>7,} dòng, {os.path.getsize(val_path) / 1024**2:.1f} MB)")
    print(f" 3. Tập TEST  : {test_path}   ({len(test_df):>7,} dòng, {os.path.getsize(test_path) / 1024**2:.1f} MB)")
    print("=" * 80)

if __name__ == '__main__':
    generate_split_csvs()
