"""
Script 3: SHAP Explainability Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

print("=" * 55)
print("  Script 3: SHAP Explainability Analysis")
print("=" * 55)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/forming_data.csv')
FEATURES    = ['MAT', 'FC', 'SHTK', 'BF']
REG_TARGETS = ['max_stress', 'thinning', 'springback']
ALL_LABELS  = ['Safe', 'Risk of Crack', 'Crack', 'Wrinkle', 'Wrinkling Tend.']
X   = df[FEATURES].values
y_r = df[REG_TARGETS].values
y_c = df['outcome'].values
unique_classes = sorted(df['outcome'].unique())
n_classes      = len(unique_classes)
OUTCOME_LABELS = [ALL_LABELS[i] for i in unique_classes]
print(f"\n  Classes: {n_classes} → {OUTCOME_LABELS}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_s   = scaler_X.fit_transform(X)
y_r_s = scaler_y.fit_transform(y_r)
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(X_s, y_r_s, y_c, test_size=0.2, random_state=42, stratify=y_c)
Xtr = torch.FloatTensor(X_tr)
Xte = torch.FloatTensor(X_te)

class SurrogateNet(nn.Module):
    def __init__(self, in_dim=4, reg_out=3, cls_out=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim,64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128,128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128,64),    nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.reg_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,reg_out))
        self.cls_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,cls_out))
    def forward(self, x):
        z = self.encoder(x)
        return self.reg_head(z), self.cls_head(z)

model = SurrogateNet(cls_out=n_classes)
model.load_state_dict(torch.load('models/best_surrogate.pt'))
model.eval()
print(f"  ✅ Model loaded")

def reg_predict(x):
    with torch.no_grad():
        reg, _ = model(torch.FloatTensor(x))
    return reg.numpy()

def cls_predict(x):
    with torch.no_grad():
        _, cls = model(torch.FloatTensor(x))
    return torch.softmax(cls, dim=1).numpy()

background  = Xtr[:200].numpy()
test_sample = Xte[:300].numpy()
print(f"\n  Computing SHAP values...")
explainer_reg = shap.KernelExplainer(reg_predict, background)
explainer_cls = shap.KernelExplainer(cls_predict, background)
shap_reg = explainer_reg.shap_values(test_sample, nsamples=100)
shap_cls = explainer_cls.shap_values(test_sample, nsamples=100)
print(f"  ✅ SHAP values computed")

def get_shap(shap_vals, idx):
    if isinstance(shap_vals, list):
        return np.array(shap_vals[idx])
    elif shap_vals.ndim == 3:
        return shap_vals[:, :, idx]
    return shap_vals

colors_feat = ['#E07B39', '#5F9EA0', '#9B59B6', '#2ECC71']
colors_reg  = ['#4C72B0', '#55A868', '#C44E52']

# Plot 1 — Regression importance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('SHAP Feature Importance — Regression Outputs', fontsize=13, fontweight='bold')
for i, (name, color) in enumerate(zip(REG_TARGETS, colors_reg)):
    sv = get_shap(shap_reg, i)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ax = axes[i]
    bars = ax.barh(range(4), mean_abs[order], color=[colors_feat[j] for j in order], alpha=0.85, height=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([FEATURES[j] for j in order], fontsize=11)
    ax.set_title(f'Impact on {name}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, mean_abs[order]):
        ax.text(val+0.0001, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/3a_shap_regression_importance.png', dpi=150, bbox_inches='tight')
print(f"  Plot → results/3a_shap_regression_importance.png")

# Plot 2 — Most critical class (crack or highest risk)
crack_idx = unique_classes.index(2) if 2 in unique_classes else unique_classes.index(1) if 1 in unique_classes else 0
crack_label = ALL_LABELS[unique_classes[crack_idx]]
sv_crack = get_shap(shap_cls, crack_idx)
mean_abs_crack = np.abs(sv_crack).mean(axis=0)
order_crack = np.argsort(mean_abs_crack)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'SHAP Analysis — {crack_label} Prediction', fontsize=13, fontweight='bold')
ax = axes[0]
bars = ax.bar(range(4), mean_abs_crack[order_crack], color=[colors_feat[j] for j in order_crack], alpha=0.85, width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([FEATURES[j] for j in order_crack], fontsize=11)
ax.set_title(f'Feature Importance for {crack_label}', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mean_abs_crack[order_crack]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.0001, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax = axes[1]
bf_orig = scaler_X.inverse_transform(test_sample)[:, 3]
sc = ax.scatter(bf_orig, sv_crack[:, 3], c=scaler_X.inverse_transform(test_sample)[:, 2], cmap='RdYlGn_r', alpha=0.5, s=15)
plt.colorbar(sc, ax=ax, label='Sheet Thickness (mm)')
ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('Blank Holder Force BF (kN)')
ax.set_ylabel(f'SHAP value')
ax.set_title('BF vs Risk (colour = SHTK)', fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/3b_shap_crack_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Plot → results/3b_shap_crack_analysis.png")

# Plot 3 — Heatmap
fig, ax = plt.subplots(figsize=(10, max(4, n_classes*1.2)))
shap_matrix = np.zeros((n_classes, 4))
for i in range(n_classes):
    sv = get_shap(shap_cls, i)
    shap_matrix[i] = np.abs(sv).mean(axis=0)
im = ax.imshow(shap_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
ax.set_xticks(range(4))
ax.set_xticklabels(FEATURES, fontsize=12)
ax.set_yticks(range(n_classes))
ax.set_yticklabels(OUTCOME_LABELS, fontsize=11)
ax.set_title('SHAP Heatmap — Feature Impact on Each Outcome', fontsize=12, fontweight='bold')
for r in range(n_classes):
    for c in range(4):
        ax.text(c, r, f'{shap_matrix[r,c]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if shap_matrix[r,c] > shap_matrix.max()*0.6 else 'black')
plt.tight_layout()
plt.savefig('results/3c_shap_heatmap_all_outcomes.png', dpi=150, bbox_inches='tight')
print(f"  Plot → results/3c_shap_heatmap_all_outcomes.png")

print(f"\n  ── Key Scientific Insights ──")
print(f"  Most influential for {crack_label}: {FEATURES[order_crack[0]]}")
for rank, idx in enumerate(order_crack):
    print(f"    {rank+1}. {FEATURES[idx]:<6}: {mean_abs_crack[idx]:.4f}")
print(f"\n✅ Script 3 complete\n")