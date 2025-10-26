# m2_transformer_seq.py
# Transformer encoder over (T, n_mfcc) sequence (AST-like idea on MFCC).

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils_mfcc import *

# DATASET_PATH      = "dataFold1.json"
# DATASET_PATH_TEST = "dataTestFold1.json"
BATCH     = 64
EPOCHS    = 30
LR        = 2e-4
NUM_CLASSES = 10
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class MFCCTransformer(nn.Module):
    def __init__(self, n_mfcc, d=128, heads=4, layers=2, num_classes=10, max_T=130):
        super().__init__()
        self.proj = nn.Linear(n_mfcc, d)
        self.pos  = nn.Parameter(torch.zeros(1, max_T, d))
        enc       = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=d*4, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls  = nn.Linear(d, num_classes)
    def forward(self, x_seq):             # (B,T,n_mfcc)
        T = x_seq.shape[1]
        z = self.proj(x_seq) + self.pos[:, :T, :]
        z = self.enc(z).mean(1)
        return self.cls(z)

# Data
(X_cnn, X_cnn_val, X_cnn_test,
 X_seq, X_seq_val, X_seq_test,
 y, y_val, y_test, n_mfcc, T) = prepare_datasets(0.2, 0.2, DATASET_PATH, DATASET_PATH_TEST)

train_dl = DataLoader(SeqMFCCDataset(X_seq, y), batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(SeqMFCCDataset(X_seq_val, y_val), batch_size=BATCH)
test_dl  = DataLoader(SeqMFCCDataset(X_seq_test, y_test), batch_size=BATCH)

# Model
model = MFCCTransformer(n_mfcc, num_classes=NUM_CLASSES, max_T=T).to(DEVICE)
crit  = nn.CrossEntropyLoss()
opt   = optim.Adam(model.parameters(), lr=LR)

# Train with timing
train_time_sum = 0.0
val_time_sum   = 0.0
for ep in range(EPOCHS):
    tr_loss, tr_acc, t_tr = train_epoch(model, train_dl, crit, opt, DEVICE)
    val_acc, t_val = evaluate(model, val_dl, DEVICE)
    train_time_sum += t_tr; val_time_sum += t_val
    print(f"Epoch {ep+1:02d} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f} "
          f"| train_time={fmt_secs(t_tr)} | val_time={fmt_secs(t_val)}")

final_train_acc, t_train_full = evaluate(model, train_dl, DEVICE)
test_acc, t_test = evaluate(model, test_dl, DEVICE)

print("\n=== Report (Transformer on MFCC sequence) ===")
print(f"Final TRAIN Accuracy : {final_train_acc:.4f} (eval_time={fmt_secs(t_train_full)})")
print(f"TEST  Accuracy       : {test_acc:.4f} (eval_time={fmt_secs(t_test)})")
print(f"Total TRAIN loop time: {fmt_secs(train_time_sum)} | Total VAL eval time: {fmt_secs(val_time_sum)}")
