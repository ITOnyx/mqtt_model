import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from mqtt_dataset import load_mqtt_dataset

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    [+] Saved confusion matrix to {save_path}")

def plot_feature_importance(importances, feature_names, top_n=20, save_path="reports/feature_importance.png"):
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f'Top {top_n} Most Important Features for MQTT Attack Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    [+] Saved feature importance plot to {save_path}")

def train_and_evaluate(args):
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    
    # 1. Load Data (Train / Validation / Test)
    data = load_mqtt_dataset(
        data_dir=args.data_dir,
        mode=args.mode,
        classification_type=args.classification,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        sample_frac=args.sample_frac
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']
    class_names = data['class_names']
    
    # 2. Define Candidate Models
    models = {}
    if args.model in ['rf', 'all']:
        models['Random_Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            random_state=args.seed,
            n_jobs=-1
        )
    if args.model in ['hgb', 'all']:
        models['Hist_Gradient_Boosting'] = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            early_stopping=False,
            random_state=args.seed
        )
    if args.model in ['dt', 'all']:
        models['Decision_Tree'] = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=20,
            random_state=args.seed
        )
        
    results = []
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    models_pbar = tqdm(models.items(), desc="[*] Overall Progress", unit="model", ncols=100)
    for name, model in models_pbar:
        models_pbar.set_description(f"[*] Training {name}")
        print(f"\n==================================================")
        print(f"[*] Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Validation Evaluation
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # Test Evaluation (Unbiased Test Set)
        infer_start = time.time()
        y_test_pred = model.predict(X_test)
        infer_time = time.time() - infer_start
        
        test_acc = accuracy_score(y_test, y_test_pred)
        bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"[*] {name} Results:")
        print(f"    - Training Time: {train_time:.2f}s | Inference Time: {infer_time:.2f}s")
        print(f"    - [Val Set]  Accuracy: {val_acc * 100:.2f}% | Macro F1: {val_macro_f1 * 100:.2f}%")
        print(f"    - [Test Set] Accuracy: {test_acc * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}% (Weighted: {weighted_f1 * 100:.2f}%)\n")
        
        report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4)
        print("Final Test Classification Report:")
        print(report)
        
        # Save Report Text
        report_file = os.path.join(args.reports_dir, f"{name.lower()}_{args.mode}_{args.classification}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Classification: {args.classification}\n")
            f.write(f"Val Accuracy: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f} | Test Macro F1: {macro_f1:.4f}\n\n")
            f.write(report)
            
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        cm_path = os.path.join(args.reports_dir, f"cm_{name.lower()}_{args.mode}_{args.classification}.png")
        plot_confusion_matrix(cm, class_names, f"Confusion Matrix: {name} ({args.mode.capitalize()})", cm_path)
        
        # Save individual model
        model_save_path = os.path.join(args.models_dir, f"{name.lower()}_{args.mode}_{args.classification}.joblib")
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'class_names': class_names,
            'mode': args.mode,
            'classification': args.classification
        }, model_save_path)
        print(f"    [+] Saved model artifact to {model_save_path}")
        
        if hasattr(model, 'feature_importances_'):
            fi_path = os.path.join(args.reports_dir, f"feature_importance_{name.lower()}_{args.mode}.png")
            plot_feature_importance(model.feature_importances_, feature_names, top_n=20, save_path=fi_path)
            
        results.append({
            'Model': name,
            'Val_Accuracy': val_acc,
            'Val_Macro_F1': val_macro_f1,
            'Test_Accuracy': test_acc,
            'Test_Macro_F1': macro_f1,
            'Test_Weighted_F1': weighted_f1,
            'Train_Time_s': train_time
        })
        
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_model_name = name
            best_model_obj = model
            
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.reports_dir, f"summary_{args.mode}_{args.classification}.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n==================================================")
    print("[*] TRAINING SUMMARY (Train / Val / Test):")
    print(summary_df.to_string(index=False))
    print(f"\n[*] Best Model (Selected by Val F1): {best_model_name} with Val Macro F1 = {best_f1 * 100:.2f}%")
    
    # Save standard best model as default for inference
    best_save_path = os.path.join(args.models_dir, f"best_mqtt_model_{args.mode}_{args.classification}.joblib")
    joblib.dump({
        'model': best_model_obj,
        'scaler': scaler,
        'feature_names': feature_names,
        'class_names': class_names,
        'mode': args.mode,
        'classification': args.classification
    }, best_save_path)
    print(f"[*] Default deployment model saved to: {best_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Machine Learning Models on MQTT Dataset")
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--mode', type=str, default='biflow', choices=['biflow', 'uniflow'])
    parser.add_argument('--classification', type=str, default='multiclass', choices=['multiclass', 'binary'])
    parser.add_argument('--model', type=str, default='all', choices=['rf', 'hgb', 'dt', 'all'])
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Fraction of data to use (1.0 = 100%)')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set fraction (default: 0.15 = 15%)')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set fraction (default: 0.15 = 15%)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--reports_dir', type=str, default='reports')
    
    args = parser.parse_args()
    train_and_evaluate(args)
