"""
Script 2: PyTorch Surrogate Model
==================================
Trains a neural network to replace expensive FEM simulations.
Given 4 process parameters → predicts stress, thinning, springback
and forming outcome (Safe / Crack / Wrinkle etc.)

This is the core surrogate modelling contribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                             confusion_matrix, r2_score)
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  Script 2: PyTorch Surrogate Model")
print("=" * 55)

df = pd.read_csv('data/forming_data.csv')
print(f"\n  Loaded {len(df):,} simulations")

FEATURES = ['MAT', 'FC', 'SHTK', 'BF']
REG_TARGETS = ['max_stress', 'thinning', 'springback']
CLS_TARGET  = 'outcome'

X   = df[FEATURES].values
y_r = df[REG_TARGETS].values   # regression
y_c = df[CLS_TARGET].values    # classification

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_s   = scaler_X.fit_transform(X)
y_r_s = scaler_y.fit_transform(y_r)

(X_tr, X_te,
 yr_tr, yr_te,
 yc_tr, yc_te) = train_test_split(
    X_s, y_r_s, y_c,
    test_size=0.2, random_state=42, stratify=y_c
)

def to_tensor(*arrays):
    return [torch.FloatTensor(a) for a in arrays]

Xtr, Xte     = to_tensor(X_tr, X_te)
yrtr, yrte   = to_tensor(yr_tr, yr_te)
yctr = torch.LongTensor(yc_tr)
ycte = torch.LongTensor(yc_te)

train_ds = TensorDataset(Xtr, yrtr, yctr)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

print(f"  Train : {len(X_tr):,}  |  Test : {len(X_te):,}")

# ─────────────────────────────────────────────
# 3. MODEL DEFINITION
# ─────────────────────────────────────────────
class SurrogateNet(nn.Module):
    """
    Multi-task surrogate network:
    - Shared encoder (process parameters → latent representation)
    - Regression head  (predicts stress, thinning, springback)
    - Classification head (predicts FLD outcome category)
    """
    def __init__(self, in_dim=4, reg_out=3, cls_out=5):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, reg_out)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cls_out)
        )

    def forward(self, x):
        z   = self.encoder(x)
        reg = self.reg_head(z)
        cls = self.cls_head(z)
        return reg, cls

n_classes  = len(np.unique(y_c))
model = SurrogateNet(cls_out=n_classes)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)
reg_loss_fn = nn.MSELoss()
cls_loss_fn = nn.CrossEntropyLoss()

EPOCHS = 200
train_losses, val_losses = [], []
best_val_loss = float('inf')

print(f"\n  Training for {EPOCHS} epochs...")
print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10}")
print(f"  {'-'*40}")

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    batch_losses = []
    for Xb, yrb, ycb in train_dl:
        optimizer.zero_grad()
        reg_pred, cls_pred = model(Xb)
        loss = reg_loss_fn(reg_pred, yrb) + 0.5 * cls_loss_fn(cls_pred, ycb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # ── Validate ──
    model.eval()
    with torch.no_grad():
        reg_pred, cls_pred = model(Xte)
        val_loss = (reg_loss_fn(reg_pred, yrte)
                    + 0.5 * cls_loss_fn(cls_pred, ycte)).item()

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_surrogate.pt')

    if (epoch + 1) % 25 == 0:
        print(f"  {epoch+1:>6} | {train_loss:>12.4f} | {val_loss:>10.4f}")

print(f"\n  ✅ Best validation loss: {best_val_loss:.4f}")
print(f"  Saved → models/best_surrogate.pt")

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
model.load_state_dict(torch.load('models/best_surrogate.pt'))
model.eval()

with torch.no_grad():
    reg_pred_s, cls_logits = model(Xte)

# Regression metrics
reg_pred = scaler_y.inverse_transform(reg_pred_s.numpy())
reg_true = scaler_y.inverse_transform(yrte.numpy())

print(f"\n  ── Regression Results ──")
for i, name in enumerate(REG_TARGETS):
    r2 = r2_score(reg_true[:, i], reg_pred[:, i])
    mae = np.mean(np.abs(reg_true[:, i] - reg_pred[:, i]))
    print(f"    {name:<15}: R² = {r2:.4f}  |  MAE = {mae:.3f}")

# Classification metrics
cls_pred = cls_logits.argmax(dim=1).numpy()
cls_true = ycte.numpy()
acc = (cls_pred == cls_true).mean()
print(f"\n  ── Classification Results ──")
print(f"    Accuracy: {acc:.4f} ({acc*100:.1f}%)")

ALL_LABELS = ['Safe', 'Risk of Crack', 'Crack', 'Wrinkle', 'Wrinkling Tend.']
unique_classes = sorted(set(cls_true) | set(cls_pred))
LABELS = [ALL_LABELS[i] for i in unique_classes]
print(f"\n{classification_report(cls_true, cls_pred, target_names=LABELS, labels=unique_classes)}")

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Surrogate Model Results', fontsize=14, fontweight='bold')

# Training curves
ax = axes[0, 0]
ax.plot(train_losses, label='Train', color='#4C72B0', lw=2)
ax.plot(val_losses,   label='Val',   color='#C44E52', lw=2)
ax.set_title('Training & Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Parity plots for each regression target
colors = ['#55A868', '#8172B2', '#CCB974']
for i, (name, color) in enumerate(zip(REG_TARGETS, colors)):
    ax = axes[0, i+1] if i < 2 else axes[1, 0]
    ax.scatter(reg_true[:, i], reg_pred[:, i],
               alpha=0.3, s=5, color=color)
    mn = min(reg_true[:, i].min(), reg_pred[:, i].min())
    mx = max(reg_true[:, i].max(), reg_pred[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect')
    r2 = r2_score(reg_true[:, i], reg_pred[:, i])
    ax.set_title(f'{name}  (R²={r2:.3f})')
    ax.set_xlabel('FEM (True)')
    ax.set_ylabel('ML (Predicted)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1, 1]
n_cls = len(unique_classes)
cm = confusion_matrix(cls_true, cls_pred, labels=unique_classes)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(n_cls))
ax.set_yticks(range(n_cls))
ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=7)
ax.set_yticklabels(LABELS, fontsize=7)
ax.set_title(f'Confusion Matrix (Acc={acc:.2f})')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
for r in range(n_cls):
    for c in range(n_cls):
        ax.text(c, r, cm[r, c], ha='center', va='center',
                fontsize=8, color='white' if cm[r, c] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)

# Outcome distribution
ax = axes[1, 2]
unique, counts = np.unique(cls_true, return_counts=True)
pred_counts = [np.sum(cls_pred == u) for u in unique]
x = np.arange(len(unique))
w = 0.35
ax.bar(x - w/2, counts,      w, label='True', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, pred_counts, w, label='Pred', color='#C44E52', alpha=0.8)
ax.set_xticks(x)
ALL_LABELS = ['Safe', 'Risk of Crack', 'Crack', 'Wrinkle', 'Wrinkling Tend.']
ax.set_xticklabels([ALL_LABELS[u] for u in unique], rotation=30,
                   ha='right', fontsize=7)
ax.set_title('True vs Predicted Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/2_surrogate_model_results.png', dpi=150, bbox_inches='tight')
print(f"\n  Plot  → results/2_surrogate_model_results.png")
print("\n✅ Script 2 complete — run 3_shap_analysis.py next\n")