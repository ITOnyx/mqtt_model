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

import numpy as np
import pandas as pd
from mqtt_dataset import load_mqtt_dataset

def display_splits_info():
    print("[*] Loading dataset and verifying Train / Val / Test split structure...")
    data = load_mqtt_dataset(mode='biflow', classification_type='multiclass', val_size=0.15, test_size=0.15)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    features = data['feature_names']
    
    print("=" * 80)
    print("                KIEM TRA TINH DOC LAP & TACH BIET GIUA CAC TAP")
    print("=" * 80)
    print(f" [+] Kich thuoc ma tran Train (X_train): {X_train.shape[0]:>7,} dong x {X_train.shape[1]} dac trung")
    print(f" [+] Kich thuoc ma tran Val   (X_val)  : {X_val.shape[0]:>7,} dong x {X_val.shape[1]} dac trung")
    print(f" [+] Kich thuoc ma tran Test  (X_test) : {X_test.shape[0]:>7,} dong x {X_test.shape[1]} dac trung")
    print(f" [+] Tong so dong du lieu:              {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:>7,} dong")
    print(f" [+] Tong so dac trung:                 {len(features)} dac trung luong mang")
    print("\n [+] NGUYEN TAC TOAN HOC & DATA SCIENCE:")
    print("     1. Train Set (70%): Dùng de hoc cac trong so / cay quyet dinh.")
    print("     2. Val Set   (15%): Dung de theo doi hoi tu tung epoch & chon checkpoint tot nhat.")
    print("     3. Test Set  (15%): Giu kin 100%, chi mang ra danh gia cuoi cung sau khi train xong.")
    print("     4. Chong Data Leakage: Scaler chi hoc tren Train Set (fit_transform tren Train duy nhat).")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    display_splits_info()
