import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from mqtt_dataset import load_mqtt_dataset

class MQTTNet(nn.Module):
    """Deep Neural Network for MQTT Intrusion Detection."""
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout=0.2):
        super(MQTTNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    [+] Saved confusion matrix to {save_path}")

def train_deep_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[*] Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
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
    num_classes = len(class_names)
    input_dim = X_train.shape[1]
    
    # Compute class weights for imbalanced handling
    classes_counts = np.bincount(y_train)
    weights = len(y_train) / (len(classes_counts) * classes_counts.astype(np.float32))
    class_weights_t = torch.FloatTensor(weights).to(device)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    # 2. Build Model
    model = MQTTNet(input_dim=input_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    print(f"[*] Initialized MQTTNet architecture with {sum(p.numel() for p in model.parameters()):,} trainable parameters.")
    
    best_macro_f1 = 0.0
    best_epoch = 1
    best_weights_path = os.path.join(args.models_dir, f"mqttnet_{args.mode}_{args.classification}.pt")
    metadata_path = os.path.join(args.models_dir, f"mqttnet_{args.mode}_{args.classification}_meta.joblib")
    patience_counter = 0
    
    print("\n[*] Starting Deep Learning Training...")
    start_time = time.time()
    
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Total Progress", unit="epoch", ncols=100)
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"  Epoch [{epoch:02d}/{args.epochs:02d}] Train", unit="batch", leave=False, ncols=100)
        for batch_X, batch_y in batch_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            running_loss += loss_val * batch_X.size(0)
            batch_pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation on validation set (Tune & Early Stopping)
        model.eval()
        val_preds = []
        val_targets = []
        val_pbar = tqdm(val_loader, desc=f"  Epoch [{epoch:02d}/{args.epochs:02d}] Val  ", unit="batch", leave=False, ncols=100)
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        scheduler.step(val_f1)
        epoch_pbar.set_postfix({'loss': f"{epoch_loss:.4f}", 'val_acc': f"{val_acc*100:.2f}%", 'val_f1': f"{val_f1*100:.2f}%"})
        print(f"    Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if args.early_stopping and patience_counter >= args.patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break
                
    total_time = time.time() - start_time
    print(f"\n[*] Training completed full {epoch} epochs in {total_time:.2f}s. Best model checkpoint was at Epoch {best_epoch} (Val Macro F1: {best_macro_f1*100:.2f}%).")
    
    # 3. Final Evaluation with best weights
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            final_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    final_preds = np.array(final_preds)
    
    report = classification_report(y_test, final_preds, target_names=class_names, digits=4)
    print("\n[*] Final Deep Model Classification Report:")
    print(report)
    
    # Save Report and Confusion Matrix
    report_file = os.path.join(args.reports_dir, f"mqttnet_{args.mode}_{args.classification}_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Model: MQTTNet (Deep Learning PyTorch)\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Classification: {args.classification}\n")
        f.write(f"Macro F1: {best_macro_f1:.4f}\n\n")
        f.write(report)
        
    cm = confusion_matrix(y_test, final_preds)
    cm_path = os.path.join(args.reports_dir, f"cm_mqttnet_{args.mode}_{args.classification}.png")
    plot_confusion_matrix(cm, class_names, f"Confusion Matrix: MQTTNet ({args.mode.capitalize()})", cm_path)
    
    # Save Metadata for Inference
    joblib.dump({
        'scaler': scaler,
        'feature_names': feature_names,
        'class_names': class_names,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'mode': args.mode,
        'classification': args.classification
    }, metadata_path)
    
    print(f"[+] Saved PyTorch Model weights: {best_weights_path}")
    print(f"[+] Saved Metadata & Scaler: {metadata_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Deep Learning (PyTorch) Model on MQTT Dataset")
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--mode', type=str, default='biflow', choices=['biflow', 'uniflow'])
    parser.add_argument('--classification', type=str, default='multiclass', choices=['multiclass', 'binary'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping (default is False: trains all epochs to the end)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--sample_frac', type=float, default=1.0)
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation fraction (default: 0.15 = 15%)')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test fraction (default: 0.15 = 15%)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--reports_dir', type=str, default='reports')
    
    args = parser.parse_args()
    train_deep_model(args)
