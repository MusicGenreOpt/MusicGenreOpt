# m1_bho_dualcnn.py
# Dual-branch CNN mimicking the MFCC/STFT branches by using
# two different kernel scales on the single MFCC input.

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils_mfcc import *

# DATASET_PATH      = "dataFold1.json"
# DATASET_PATH_TEST = "dataTestFold1.json"
BATCH     = 64
EPOCHS    = 30
LR        = 1e-3
NUM_CLASSES = 10
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class CNNBranch(nn.Module):
    def __init__(self, k1=3, k2=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (k1, k1), padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (k2, k2), padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):  # (B,1,n_mfcc,T)
        return self.net(x).flatten(1)  # (B,128)

class DualCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.b1 = CNNBranch(k1=3, k2=5)
        self.b2 = CNNBranch(k1=5, k2=7)
        self.fc = nn.Linear(128 * 2, num_classes)
    def forward(self, x):
        f = torch.cat([self.b1(x), self.b2(x)], dim=1)
        return self.fc(f)

# Data
(X_cnn, X_cnn_val, X_cnn_test,
 _, _, _,
 y, y_val, y_test, _, _) = prepare_datasets(0.2, 0.2, DATASET_PATH, DATASET_PATH_TEST)

train_dl = DataLoader(CNNMFCCDataset(X_cnn, y), batch_size=BATCH, shuffle=True, drop_last=False)
val_dl   = DataLoader(CNNMFCCDataset(X_cnn_val, y_val), batch_size=BATCH, shuffle=False)
test_dl  = DataLoader(CNNMFCCDataset(X_cnn_test, y_test), batch_size=BATCH, shuffle=False)

# Model
model = DualCNN(NUM_CLASSES).to(DEVICE)
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

# Final TRAIN / VAL / TEST
final_train_acc, t_train_full = evaluate(model, train_dl, DEVICE)
final_val_acc,   t_val_full   = evaluate(model, val_dl,   DEVICE)
test_acc,        t_test       = evaluate(model, test_dl,  DEVICE)

print("\n=== Final Report (DualCNN) ===")
print(f"Final TRAIN Accuracy : {final_train_acc:.4f} (eval_time={fmt_secs(t_train_full)})")
print(f"Final VAL   Accuracy : {final_val_acc:.4f} (eval_time={fmt_secs(t_val_full)})")
print(f"TEST  Accuracy       : {test_acc:.4f} (eval_time={fmt_secs(t_test)})")
print(f"Total TRAIN loop time: {fmt_secs(train_time_sum)} | Total VAL eval time (per-epoch sum): {fmt_secs(val_time_sum)}")
