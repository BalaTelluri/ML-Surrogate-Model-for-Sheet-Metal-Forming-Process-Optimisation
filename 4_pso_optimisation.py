"""
Script 4: PSO Optimisation
============================
Uses the trained surrogate model as an objective function to find
optimal process parameters that minimise crack/wrinkle probability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

print("=" * 55)
print("  Script 4: PSO Process Optimisation")
print("=" * 55)

# ─────────────────────────────────────────────
# 1. RELOAD MODEL + SCALERS
# ─────────────────────────────────────────────
df = pd.read_csv('data/forming_data.csv')
FEATURES    = ['MAT', 'FC', 'SHTK', 'BF']
REG_TARGETS = ['max_stress', 'thinning', 'springback']

X   = df[FEATURES].values
y_r = df[REG_TARGETS].values
y_c = df['outcome'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_s   = scaler_X.fit_transform(X)
y_r_s = scaler_y.fit_transform(y_r)

X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
    X_s, y_r_s, y_c, test_size=0.2, random_state=42, stratify=y_c
)

class SurrogateNet(nn.Module):
    def __init__(self, in_dim=4, reg_out=3, cls_out=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 128),     nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.reg_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, reg_out))
        self.cls_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, cls_out))

    def forward(self, x):
        z = self.encoder(x)
        return self.reg_head(z), self.cls_head(z)

# Fix cls_out to match actual number of classes
n_classes = len(np.unique(y_c))
model = SurrogateNet(cls_out=n_classes)
model.load_state_dict(torch.load('models/best_surrogate.pt'), strict=False)
model.eval()
print(f"\n  ✅ Model loaded")

# ─────────────────────────────────────────────
# 2. OBJECTIVE FUNCTION
# ─────────────────────────────────────────────
SAFE_IDX  = 0
CRACK_IDX = min(2, n_classes - 1)

def objective(params_raw):
    params_scaled = scaler_X.transform(params_raw)
    t = torch.FloatTensor(params_scaled)
    with torch.no_grad():
        reg_s, cls_logits = model(t)
    probs    = torch.softmax(cls_logits, dim=1).numpy()
    reg_vals = scaler_y.inverse_transform(reg_s.numpy())
    crack_prob = probs[:, CRACK_IDX] if probs.shape[1] > CRACK_IDX else np.zeros(len(probs))
    safe_prob  = probs[:, SAFE_IDX]
    thinning   = reg_vals[:, 1]
    fitness = 2.0 * crack_prob + 0.3 * thinning - 1.0 * safe_prob
    return fitness

# ─────────────────────────────────────────────
# 3. PSO
# ─────────────────────────────────────────────
BOUNDS = np.array([
    [0.90, 1.10],
    [0.05, 0.15],
    [0.95, 1.00],
    [df['BF'].min(), df['BF'].max()],
])

N_PARTICLES = 50
N_DIMS      = 4
MAX_ITER    = 100
W, C1, C2   = 0.7, 1.5, 1.5

print(f"\n  Running PSO: {N_PARTICLES} particles, {MAX_ITER} iterations")

np.random.seed(0)
pos = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (N_PARTICLES, N_DIMS))
vel = np.random.uniform(-0.1, 0.1, (N_PARTICLES, N_DIMS))

p_best_pos = pos.copy()
p_best_fit = objective(pos)
g_best_idx = np.argmin(p_best_fit)
g_best_pos = p_best_pos[g_best_idx].copy()
g_best_fit = p_best_fit[g_best_idx]

history_best   = []
history_mean   = []
history_params = []

print(f"\n  {'Iter':>5} | {'Best Fitness':>14} | {'Mean Fitness':>13}")
print(f"  {'-'*40}")

for it in range(MAX_ITER):
    r1 = np.random.rand(N_PARTICLES, N_DIMS)
    r2 = np.random.rand(N_PARTICLES, N_DIMS)
    vel = W * vel + C1 * r1 * (p_best_pos - pos) + C2 * r2 * (g_best_pos - pos)
    pos = np.clip(pos + vel, BOUNDS[:, 0], BOUNDS[:, 1])
    fit = objective(pos)

    improved = fit < p_best_fit
    p_best_pos[improved] = pos[improved]
    p_best_fit[improved] = fit[improved]

    new_best_idx = np.argmin(p_best_fit)
    if p_best_fit[new_best_idx] < g_best_fit:
        g_best_fit = p_best_fit[new_best_idx]
        g_best_pos = p_best_pos[new_best_idx].copy()

    history_best.append(g_best_fit)
    history_mean.append(fit.mean())
    history_params.append(g_best_pos.copy())

    if (it + 1) % 20 == 0:
        print(f"  {it+1:>5} | {g_best_fit:>14.4f} | {fit.mean():>13.4f}")

# ─────────────────────────────────────────────
# 4. EVALUATE OPTIMAL
# ─────────────────────────────────────────────
opt_params = g_best_pos.reshape(1, -1)
opt_scaled = scaler_X.transform(opt_params)

with torch.no_grad():
    reg_s, cls_logits = model(torch.FloatTensor(opt_scaled))

opt_probs = torch.softmax(cls_logits, dim=1).numpy()[0]
opt_reg   = scaler_y.inverse_transform(reg_s.numpy())[0]

ALL_LABELS = ['Safe', 'Risk of Crack', 'Crack', 'Wrinkle', 'Wrinkling Tend.']
LABELS = ALL_LABELS[:n_classes]

print(f"\n  ── Optimal Parameters ──")
for feat, val in zip(FEATURES, g_best_pos):
    print(f"    {feat:<6}: {val:.4f}")

print(f"\n  ── Predicted Outcomes ──")
print(f"    Max Stress : {opt_reg[0]:.2f} MPa")
print(f"    Thinning   : {opt_reg[1]:.4f}")
print(f"    Springback : {opt_reg[2]:.2f}°")

print(f"\n  ── FLD Probabilities ──")
for label, prob in zip(LABELS, opt_probs):
    bar = '█' * int(prob * 30)
    print(f"    {label:<22}: {prob:.3f}  {bar}")

# Save
opt_df = pd.DataFrame({
    'Parameter':     FEATURES,
    'Optimal_Value': g_best_pos,
    'Unit':          ['factor', 'coefficient', 'mm', 'kN']
})
opt_df.to_csv('results/optimal_parameters.csv', index=False)
print(f"\n  Saved → results/optimal_parameters.csv")

# ─────────────────────────────────────────────
# 5. PLOTS — Clean simple version
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('PSO Optimisation Results', fontsize=14, fontweight='bold')

# 1. Convergence
axes[0, 0].plot(history_best, color='#C44E52', lw=2, label='Best')
axes[0, 0].plot(history_mean, color='#4C72B0', lw=1.5, ls='--', alpha=0.7, label='Mean')
axes[0, 0].set_title('PSO Convergence')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Fitness')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. MAT convergence
axes[0, 1].plot([h[0] for h in history_params], color='#E07B39', lw=2)
axes[0, 1].set_title('MAT Parameter Convergence')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('MAT value')
axes[0, 1].grid(True, alpha=0.3)

# 3. BF convergence
axes[0, 2].plot([h[3] for h in history_params], color='#2ECC71', lw=2)
axes[0, 2].set_title('BF Parameter Convergence')
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('BF value')
axes[0, 2].grid(True, alpha=0.3)

# 4. All params normalised
colors_p = ['#E07B39', '#5F9EA0', '#9B59B6', '#2ECC71']
for i, (feat, color) in enumerate(zip(FEATURES, colors_p)):
    vals = [(h[i] - BOUNDS[i, 0]) / (BOUNDS[i, 1] - BOUNDS[i, 0])
            for h in history_params]
    axes[1, 0].plot(vals, color=color, lw=1.5, label=feat, alpha=0.8)
axes[1, 0].set_title('All Parameters (Normalised)')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Normalised value')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Outcome probabilities
bar_colors = ['#2ECC71', '#F39C12', '#E74C3C', '#3498DB', '#9B59B6']
bars = axes[1, 1].bar(range(len(opt_probs)), opt_probs,
                      color=bar_colors[:len(opt_probs)],
                      alpha=0.85, edgecolor='white')
axes[1, 1].set_xticks(range(len(opt_probs)))
axes[1, 1].set_xticklabels(LABELS, rotation=25, ha='right', fontsize=8)
axes[1, 1].set_title('Outcome Probabilities at Optimal Params')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, opt_probs):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

# 6. Summary table
axes[1, 2].axis('off')
table_data = [['Parameter', 'Optimal', 'Unit']]
units = ['factor', 'coeff.', 'mm', 'kN']
for feat, val, unit in zip(FEATURES, g_best_pos, units):
    table_data.append([feat, f'{val:.4f}', unit])
table_data.append(['', '', ''])
table_data.append(['Max Stress', f'{opt_reg[0]:.1f}', 'MPa'])
table_data.append(['Thinning',   f'{opt_reg[1]:.4f}', '-'])
table = axes[1, 2].table(cellText=table_data[1:],
                          colLabels=table_data[0],
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#F8F9FA')
axes[1, 2].set_title('Optimal Parameter Summary', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/4_pso_optimisation_results.png', dpi=150, bbox_inches='tight')
print(f"  Plot  → results/4_pso_optimisation_results.png")
print(f"\n✅ Script 4 complete\n")
