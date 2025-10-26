# m4_improved_vit_mfcc.py
# ViT with small patches (8x8) + simple feature-wise (channel) attention (SE-like).

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils_mfcc import *

# DATASET_PATH      = "dataFold1.json"
# DATASET_PATH_TEST = "dataTestFold1.json"
BATCH     = 48
EPOCHS    = 30
LR        = 1e-4
NUM_CLASSES = 10
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class SE(nn.Module):
    """Simple Squeeze-Excitation over token features."""
    def __init__(self, d, r=8):
        super().__init__()
        self.fc1 = nn.Linear(d, d // r)
        self.fc2 = nn.Linear(d // r, d)
    def forward(self, x):  # (B, N, D)
        s = x.mean(1)  # (B, D)
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * w.unsqueeze(1)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, emb=64, ph=8, pw=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, emb, kernel_size=(ph, pw), stride=(ph, pw))
    def forward(self, x):  # (B,1,n_mfcc,T)
        z = self.proj(x)    # (B,emb,H',W')
        B, E, H, W = z.shape
        return z.flatten(2).transpose(1, 2)  # (B, N, E)

class ImprovedViT(nn.Module):
    def __init__(self, num_classes, emb=64, heads=4, layers=6):
        super().__init__()
        self.patch = PatchEmbed(1, emb, ph=8, pw=8)  # smaller patches preserve detail
        self.pos   = None
        layer      = nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=emb*4, batch_first=True)
        self.enc   = nn.TransformerEncoder(layer, num_layers=layers)
        self.se    = SE(emb)
        self.cls   = nn.Linear(emb, num_classes)
    def forward(self, x):
        z = self.patch(x)  # (B,N,E)
        if (self.pos is None) or (self.pos.shape[1] != z.shape[1]):
            self.pos = nn.Parameter(torch.zeros(1, z.shape[1], z.shape[2], device=z.device))
        z = z + self.pos
        z = self.enc(z)
        z = self.se(z)
        z = z.mean(1)
        return self.cls(z)

# Data
(X_cnn, X_cnn_val, X_cnn_test,
 _, _, _,
 y, y_val, y_test, _, _) = prepare_datasets(0.2, 0.2, DATASET_PATH, DATASET_PATH_TEST)

train_dl = DataLoader(CNNMFCCDataset(X_cnn, y), batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(CNNMFCCDataset(X_cnn_val, y_val), batch_size=BATCH)
test_dl  = DataLoader(CNNMFCCDataset(X_cnn_test, y_test), batch_size=BATCH)

# Model
model = ImprovedViT(NUM_CLASSES).to(DEVICE)
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

print("\n=== Report (Improved ViT MFCC) ===")
print(f"Final TRAIN Accuracy : {final_train_acc:.4f} (eval_time={fmt_secs(t_train_full)})")
print(f"TEST  Accuracy       : {test_acc:.4f} (eval_time={fmt_secs(t_test)})")
print(f"Total TRAIN loop time: {fmt_secs(train_time_sum)} | Total VAL eval time: {fmt_secs(val_time_sum)}")
