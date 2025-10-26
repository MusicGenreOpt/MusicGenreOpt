# utils_mfcc.py
# -----------------------------------------------
# Shared utilities for MFCC JSON loading, dataset
# prep, PyTorch Datasets, training/eval loops, and
# timing/format helpers.
# -----------------------------------------------

import json, numpy as np, time
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# ------------------------------
# JSON -> numpy tensors
# JSON format: {
#   "mapping": [...],
#   "mfcc": [ [T x n_mfcc], ... ],
#   "labels": [int, ...]
# }
# We create two shapes:
#  - CNN view: (N, 1, n_mfcc, T)
#  - SEQ view: (N, T, n_mfcc)
# ------------------------------
def load_data(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    X = [np.array(m, dtype=np.float32) for m in data["mfcc"]]   # list of (T, n_mfcc)
    y = np.array(data["labels"], dtype=np.int64)

    max_T = max(x.shape[0] for x in X)
    n_mfcc = X[0].shape[1]
    # Cap/pad time dimension to a fixed length (130 frames typical for 30s/512 hop)
    T = min(max_T, 130)

    X_cnn = np.zeros((len(X), 1, n_mfcc, T), dtype=np.float32)
    X_seq = np.zeros((len(X), T, n_mfcc), dtype=np.float32)
    for i, m in enumerate(X):
        m = m[:T] if m.shape[0] >= T else np.pad(m, ((0, T - m.shape[0]), (0, 0)))
        X_cnn[i, 0] = m.T     # (n_mfcc, T) for CNN
        X_seq[i]    = m       # (T, n_mfcc) for Transformer/sequence models
    return X_cnn, X_seq, y, n_mfcc, T

# ------------------------------
# Matches your required signature (+ test JSON path).
# Returns both CNN and SEQ views for train/val/test.
# ------------------------------
def prepare_datasets(test_size, val_size, DATASET_PATH, DATASET_PATH_TEST):
    X_cnn_all, X_seq_all, y_all, n_mfcc, T = load_data(DATASET_PATH)
    X_cnn_test, X_seq_test, y_test, _, _   = load_data(DATASET_PATH_TEST)

    # Split only the training JSON into train/val (stratified)
    X_cnn, X_cnn_val, X_seq, X_seq_val, y, y_val = train_test_split(
        X_cnn_all, X_seq_all, y_all, test_size=val_size, stratify=y_all, random_state=42
    )

    return (X_cnn, X_cnn_val, X_cnn_test,
            X_seq, X_seq_val, X_seq_test,
            y, y_val, y_test, n_mfcc, T)

# ------------------------------
# PyTorch Dataset wrappers
# ------------------------------
class CNNMFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)  # (N,1,n_mfcc,T)
        self.y = torch.tensor(y)  # (N,)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SeqMFCCDataset(Dataset):
    def __init__(self, Xseq, y):
        self.X = torch.tensor(Xseq)  # (N,T,n_mfcc)
        self.y = torch.tensor(y)     # (N,)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ------------------------------
# Generic training/evaluation with timing
# ------------------------------
def train_epoch(model, loader, criterion, optim, device):
    t0 = time.perf_counter()
    model.train()
    total = 0; correct = 0; loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * len(xb)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += len(xb)
    elapsed = time.perf_counter() - t0
    return (loss_sum / max(total, 1)), (correct / max(total, 1)), elapsed

@torch.no_grad()
def evaluate(model, loader, device):
    t0 = time.perf_counter()
    model.eval()
    total = 0; correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += len(xb)
    elapsed = time.perf_counter() - t0
    return (correct / max(total, 1)), elapsed

def fmt_secs(s: float) -> str:
    # Returns mm:ss.mmm formatting
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:06.3f}"