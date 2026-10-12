import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES_MULTICLASS = [
    'Normal',
    'Scan_A',
    'Scan_sU',
    'Sparta',
    'MQTT_Bruteforce'
]

CLASS_NAMES_BINARY = [
    'Normal',
    'Attack'
]

FILE_PREFIX_MAP = {
    'biflow': 'biflow_',
    'uniflow': 'uniflow_',
    'packet': ''
}

ATTACK_CONFIG = [
    ('normal', 0, 0),          # file_key, is_attack_filter, multiclass_label
    ('scan_A', 1, 1),
    ('scan_sU', 1, 2),
    ('sparta', 1, 3),
    ('mqtt_bruteforce', 1, 4)
]

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_split_summary(y_train, y_val, y_test, class_names):
    """Print a clear and transparent breakdown table of Train / Val / Test sets."""
    total = len(y_train) + len(y_val) + len(y_test)
    print("\n" + "=" * 90)
    print("              BANG PHAN CHIA CHI TIET TAP TRAIN / VAL / TEST (STRATIFIED)")
    print("=" * 90)
    print(f" {'ID':<3} | {'Traffic Class':<18} | {'Train Set (70%)':<17} | {'Val Set (15%)':<17} | {'Test Set (15%)':<17} | {'Total':<10}")
    print("-" * 90)
    for idx, cname in enumerate(class_names):
        n_tr = int(np.sum(y_train == idx))
        n_va = int(np.sum(y_val == idx))
        n_te = int(np.sum(y_test == idx))
        n_tot = n_tr + n_va + n_te
        p_tr = (n_tr / n_tot * 100) if n_tot else 0
        p_va = (n_va / n_tot * 100) if n_tot else 0
        p_te = (n_te / n_tot * 100) if n_tot else 0
        print(f" {idx:<3} | {cname:<18} | {n_tr:>7,} ({p_tr:5.1f}%)   | {n_va:>7,} ({p_va:5.1f}%)   | {n_te:>7,} ({p_te:5.1f}%)   | {n_tot:>10,}")
    print("-" * 90)
    p_tr_all = len(y_train) / total * 100
    p_va_all = len(y_val) / total * 100
    p_te_all = len(y_test) / total * 100
    print(f" {'TONG CONG / TOTAL':<24} | {len(y_train):>7,} ({p_tr_all:5.1f}%)   | {len(y_val):>7,} ({p_va_all:5.1f}%)   | {len(y_test):>7,} ({p_te_all:5.1f}%)   | {total:>10,}")
    print("=" * 90 + "\n")

def load_mqtt_dataset(data_dir=".", mode="biflow", classification_type="multiclass", test_size=0.15, val_size=0.15, random_state=42, sample_frac=1.0):
    """
    Load and preprocess MQTT Intrusion Detection Dataset.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset CSV files.
    mode : str
        'biflow' (default), 'uniflow', or 'packet'.
    classification_type : str
        'multiclass' (0-4) or 'binary' (0 vs 1).
    test_size : float
        Proportion of dataset for testing.
    random_state : int
        Random seed for reproducibility.
    sample_frac : float
        Fraction of data to sample (1.0 for all data, < 1.0 for faster debug).
        
    Returns:
    --------
    dict containing:
        - X_train, X_test (numpy arrays, scaled)
        - y_train, y_test (numpy arrays)
        - scaler (fitted StandardScaler)
        - feature_names (list of str)
        - class_names (list of str)
        - raw_train_df, raw_test_df
    """
    # 1. Check if standalone split files exist
    splits_candidates = [
        os.path.join(data_dir, "data", "splits"),
        os.path.join(data_dir, "splits"),
        os.path.join("data", "splits"),
        "splits"
    ]
    splits_dir = next((p for p in splits_candidates if os.path.exists(os.path.join(p, "train.csv"))), None)
    
    if mode == 'biflow' and splits_dir:
        train_csv = os.path.join(splits_dir, "train.csv")
        val_csv = os.path.join(splits_dir, "val.csv")
        test_csv = os.path.join(splits_dir, "test.csv")
        
        print(f"[*] Reading directly from standalone files in '{splits_dir}':")
        print(f"    - Train : {train_csv}")
        print(f"    - Val   : {val_csv}")
        print(f"    - Test  : {test_csv}")
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        
        if sample_frac < 1.0:
            train_df = train_df.sample(frac=sample_frac, random_state=random_state)
            val_df = val_df.sample(frac=sample_frac, random_state=random_state)
            test_df = test_df.sample(frac=sample_frac, random_state=random_state)
            
        non_feat = ['is_attack', 'target', 'class_name', 'target_multiclass', 'target_binary']
        feature_cols = [c for c in train_df.columns if c not in non_feat]
        
        if classification_type == "binary":
            y_train = train_df['is_attack'].values
            y_val = val_df['is_attack'].values
            y_test = test_df['is_attack'].values
            class_names = CLASS_NAMES_BINARY
        else:
            y_train = train_df['target'].values
            y_val = val_df['target'].values
            y_test = test_df['target'].values
            class_names = CLASS_NAMES_MULTICLASS
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_val = scaler.transform(val_df[feature_cols])
        X_test = scaler.transform(test_df[feature_cols])
        
        print_split_summary(y_train, y_val, y_test, class_names)
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': feature_cols,
            'class_names': class_names,
            'X_test_raw': test_df[feature_cols]
        }

    prefix = FILE_PREFIX_MAP.get(mode, 'biflow_')
    dfs = []
    print(f"[*] Loading dataset in mode: '{mode}', classification: '{classification_type}'...")
    data_pbar = tqdm(ATTACK_CONFIG, desc="[*] Reading dataset", ncols=100, unit="class")
    
    for file_key, attack_filter, multi_label in data_pbar:
        filename = f"{prefix}{file_key}.csv"
        candidate_paths = [
            os.path.join(data_dir, mode, filename),
            os.path.join(data_dir, filename),
            os.path.join("data", mode, filename),
            os.path.join("data", "packets", filename),
            os.path.join("data", filename),
            os.path.join(".", filename)
        ]
        filepath = None
        for p in candidate_paths:
            if os.path.exists(p):
                filepath = p
                break
                
        if not filepath:
            raise FileNotFoundError(f"Dataset file not found: {filename} in {candidate_paths}")
            
        data_pbar.set_postfix({'class': CLASS_NAMES_MULTICLASS[multi_label]})
        df = pd.read_csv(filepath)
        
        # Filter matching packets/flows
        df = df[df['is_attack'] == attack_filter].copy()
        
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
            
        df['target_multiclass'] = multi_label
        df['target_binary'] = 0 if multi_label == 0 else 1
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"[*] Total dataset size: {len(combined_df):,} samples.")
    
    # Identify non-feature columns
    cols_to_drop = [
        'ip_src', 'ip_dst', 'src_ip', 'dst_ip',
        'proto', 'protocol', 'timestamp',
        'is_attack', 'target_multiclass', 'target_binary'
    ]
    
    feature_cols = [c for c in combined_df.columns if c not in cols_to_drop]
    
    # Handle missing / infinite values if any
    X_raw = combined_df[feature_cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.fillna(-1.0)
    
    # Select target
    if classification_type == "binary":
        y_raw = combined_df['target_binary'].values
        class_names = CLASS_NAMES_BINARY
    else:
        y_raw = combined_df['target_multiclass'].values
        class_names = CLASS_NAMES_MULTICLASS
        
    # Stratified Split: Train / Validation / Test
    total_samples = len(X_raw)
    if val_size > 0.0:
        # Step 1: Split out Test set
        X_temp_df, X_test_df, y_temp, y_test = train_test_split(
            X_raw, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )
        # Step 2: Split remaining into Train and Validation sets
        val_relative_size = val_size / (1.0 - test_size)
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_temp_df, y_temp,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=y_temp
        )
    else:
        # 2-way split fallback if val_size == 0
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_raw, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )
        X_val_df, y_val = X_test_df, y_test

    # Standardize numerical features (Fit ONLY on Train to prevent data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)
    
    print_split_summary(y_train, y_val, y_test, class_names)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_cols,
        'class_names': class_names,
        'X_test_raw': X_test_df
    }

if __name__ == '__main__':
    data = load_mqtt_dataset(mode='biflow', sample_frac=0.1)
    print("Dataset module self-check passed successfully.")
