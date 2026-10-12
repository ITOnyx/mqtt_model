#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed and Modernized Classification Script for MQTT Intrusion Detection Dataset
Compatible with Python 3.10+ and Scikit-Learn 1.2+
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import argparse
import time

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

one_hot_encoder = None

def load_file(path, mode, is_attack=1, label=1, folder_name='output/', sliceno=0, verbose=True):
    global one_hot_encoder
    
    columns_to_drop_packet = ['timestamp', 'src_ip', 'dst_ip']
    # FIX: Drop is_attack to prevent label leakage in features
    columns_to_drop_uni = ['proto', 'ip_src', 'ip_dst', 'is_attack']
    columns_to_drop_bi = ['proto', 'ip_src', 'ip_dst', 'is_attack']
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
        
    dataset = pd.read_csv(path)
    
    if mode == 1 or mode == 2:
        dataset = dataset.loc[dataset['is_attack'] == is_attack]
        dataset.drop(columns=[col for col in (columns_to_drop_uni if mode == 1 else columns_to_drop_bi) if col in dataset.columns], inplace=True)
    elif mode == 0:
        dataset = dataset.loc[dataset['is_attack'] == is_attack]
        dataset.drop(columns=[col for col in columns_to_drop_packet if col in dataset.columns], inplace=True)
        dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='mqtt')))]
        if 'is_attack' in dataset.columns:
            dataset.drop('is_attack', axis=1, inplace=True)
            
    dataset = dataset.fillna(-1)
    
    if mode == 0:
        if one_hot_encoder is None:
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            first_col = dataset.iloc[:, [0]]
            encoded = one_hot_encoder.fit_transform(first_col)
        else:
            first_col = dataset.iloc[:, [0]]
            encoded = one_hot_encoder.transform(first_col)
        rest = dataset.iloc[:, 1:].values
        x = np.hstack((encoded, rest))
    else:
        x = dataset.values
        
    x = np.unique(x, axis=0)
    
    # Subsampling / slicing if dataset is huge
    if (mode == 1 and x.shape[0] > 100000) or (mode == 2 and x.shape[0] > 50000):
        temp = x.shape[0] // 10
        start = sliceno * temp
        end = min(start + temp, x.shape[0])
        x = x[start:end, :]
    elif mode == 0 and x.shape[0] > 100000:
        temp = x.shape[0] // 10
        start = sliceno * temp
        end = min(start + temp, x.shape[0])
        x = x[start:end, :]
        
    y = np.full(x.shape[0], label)
    
    if verbose:
        print(f"Loaded {os.path.basename(path)} -> Class {label}: {x.shape[0]} samples")
        
    return x, y

def classify_sub(classifier, x_train, y_train, x_test, y_test, cm_file_name, summary_file_name, classifier_name, verbose=True):
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)
    
    classifier.fit(x_train_sc, y_train)
    pred = classifier.predict(x_test_sc)
    
    cm = pd.crosstab(y_test, pred)
    cm.to_csv(cm_file_name)
    
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).transpose().to_csv(summary_file_name)
    
    if verbose:
        print(f"    - {classifier_name} Done.")

def classify(random_state, x_train, y_train, x_test, y_test, folder_name, prefix="", verbose=True):
    confusion_matrix_folder = os.path.join(folder_name, 'Confusion_Matrix') 
    summary_folder = os.path.join(folder_name, 'Summary') 

    os.makedirs(confusion_matrix_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)
    
    models = [
        ('Linear', LogisticRegression(random_state=random_state, max_iter=500)),
        ('KNN', KNeighborsClassifier()),
        ('Naive', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier(criterion='entropy', random_state=random_state)),
        ('Random Forest', RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=random_state, n_jobs=-1)),
        ('Linear SVM', LinearSVC(random_state=random_state, max_iter=2000))
    ]
    
    for name, clf in models:
        cm_file = os.path.join(confusion_matrix_folder, f"{prefix}_cm_{name.lower().replace(' ', '_')}.csv")
        sum_file = os.path.join(summary_folder, f"{prefix}_summary_{name.lower().replace(' ', '_')}.csv")
        classify_sub(clf, x_train, y_train, x_test, y_test, cm_file, sum_file, name, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='..', help='Directory where CSV files are stored')
    parser.add_argument('--mode', type=int, default=2, help='0: packet, 1: uniflow, 2: biflow')
    parser.add_argument('--output', default='Classification_Bi')
    parser.add_argument('--slices', type=int, default=1, help='Number of slices to evaluate')
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()
    
    prefix = ''
    if args.mode == 1:
        prefix = 'uniflow_' 
    elif args.mode == 2:
        prefix = 'biflow_'
        
    for slice_number in range(args.slices):
        if args.verbose:
            print(f"\n==========================================")
            print(f"Starting Slice #: {slice_number}")
            
        random_state = 0
        folder_name = f"{args.output}_{slice_number}"
        os.makedirs(folder_name, exist_ok=True)
        
        files_config = [
            (f"{prefix}normal.csv", 0, 0),
            (f"{prefix}scan_A.csv", 1, 1),
            (f"{prefix}scan_sU.csv", 1, 2),
            (f"{prefix}sparta.csv", 1, 3),
            (f"{prefix}mqtt_bruteforce.csv", 1, 4)
        ]
        
        x_list = []
        y_list = []
        
        mode_subfolder = 'biflow' if args.mode == 2 else ('uniflow' if args.mode == 1 else 'packets')
        for fname, is_att, lbl in files_config:
            candidates = [
                os.path.join(args.data_dir, 'data', mode_subfolder, fname),
                os.path.join(args.data_dir, mode_subfolder, fname),
                os.path.join(args.data_dir, fname),
                os.path.join('..', 'data', mode_subfolder, fname),
                os.path.join('data', mode_subfolder, fname),
                fname
            ]
            fpath = next((p for p in candidates if os.path.exists(p)), None)
            if not fpath:
                raise FileNotFoundError(f"Could not find {fname} in {candidates}")
            x_f, y_f = load_file(fpath, args.mode, is_att, lbl, folder_name, slice_number, args.verbose)
            x_list.append(x_f)
            y_list.append(y_f)
            
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        
        print("\nEvaluating Classifiers (Train / Test Split):")
        classify(random_state, x_train, y_train, x_test, y_test, folder_name, f"slice_{slice_number}", args.verbose)
        print(f"Slice #{slice_number} completed successfully. Outputs saved in {folder_name}/")
