"""
Script 1: Data Loader
======================
Loads real metadata.csv from DDACS dataset if available.
Falls back to synthetic data that mirrors exact DDACS structure.

TO USE YOUR REAL DATA:
  Just drop your metadata.csv into the data/ folder.
  This script automatically detects and uses it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('data',    exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('rag_db',  exist_ok=True)

print("=" * 55)
print("  Script 1: Data Loader")
print("=" * 55)

# ─────────────────────────────────────────────
# AUTO-DETECT REAL vs SYNTHETIC DATA
# ─────────────────────────────────────────────
REAL_DATA_PATH = 'data/metadata.csv'

if os.path.exists(REAL_DATA_PATH):
    print(f"\n  ✅ Real metadata.csv found — loading real data")
    df_raw = pd.read_csv(REAL_DATA_PATH)
    print(f"  Shape: {df_raw.shape}")
    print(f"  Columns: {df_raw.columns.tolist()}")

    # ── Flexible column mapping ──
    # Maps common DDACS column names to standard names
    col_map = {}
    for col in df_raw.columns:
        cl = col.lower()
        if 'mat'  in cl and 'mat' not in col_map:  col_map[col] = 'MAT'
        if 'fc'   in cl or 'fric' in cl:           col_map[col] = 'FC'
        if 'shtk' in cl or 'thick' in cl:          col_map[col] = 'SHTK'
        if 'bf'   in cl or 'binder' in cl or 'blank' in cl: col_map[col] = 'BF'
        if 'geo'  in cl:                            col_map[col] = 'geometry'
        if 'outcome' in cl or 'result' in cl or 'fld' in cl: col_map[col] = 'outcome'
        if 'stress' in cl:                          col_map[col] = 'max_stress'
        if 'thin' in cl:                            col_map[col] = 'thinning'
        if 'spring' in cl:                          col_map[col] = 'springback'

    df = df_raw.rename(columns=col_map)
    print(f"\n  Mapped columns: {col_map}")

else:
    print(f"\n  ⚠️  No metadata.csv found in data/")
    print(f"  → Generating synthetic data mirroring DDACS structure")
    print(f"  → Drop your metadata.csv into data/ to use real data\n")

    np.random.seed(42)
    N = 5000

    MAT  = np.random.uniform(0.90, 1.10, N)
    FC   = np.random.uniform(0.05, 0.15, N)
    SHTK = np.random.uniform(0.95, 1.00, N)
    BF   = np.random.uniform(100,  500,  N)
    GEOMETRY = np.random.choice(['Concave', 'Convex', 'Rectangular'], N)

    max_stress = (200 + 150*(BF/500) - 100*(SHTK/1.0)
                  + 80*MAT + 30*(FC/0.15)
                  + np.random.normal(0, 12, N))

    thinning = (0.05 + 0.12*(BF/500) - 0.06*(SHTK/1.0)
                + 0.03*MAT - 0.02*(FC/0.15)
                + np.random.normal(0, 0.005, N)).clip(0, 0.4)

    springback = (5.0 - 3.0*(BF/500) + 2.0*(SHTK/1.0)
                  + 1.5*MAT - 1.0*(FC/0.15)
                  + np.random.normal(0, 0.3, N)).clip(0, 15)

    crack        = (BF > 420) & (SHTK < 0.965) & (MAT > 1.05)
    crack_risk   = (BF > 350) & (SHTK < 0.975) & ~crack
    wrinkle      = (BF < 180) & (FC > 0.12)
    wrinkle_tend = (BF < 220) & (FC > 0.10) & ~wrinkle

    outcome = np.zeros(N, dtype=int)
    outcome[wrinkle_tend] = 4
    outcome[wrinkle]      = 3
    outcome[crack_risk]   = 1
    outcome[crack]        = 2

    OUTCOME_LABELS = {0:'Safe', 1:'Risk of Crack', 2:'Crack',
                      3:'Wrinkle', 4:'Wrinkling Tendency'}

    df = pd.DataFrame({
        'MAT': MAT, 'FC': FC, 'SHTK': SHTK, 'BF': BF,
        'geometry': GEOMETRY,
        'max_stress': max_stress,
        'thinning': thinning,
        'springback': springback,
        'outcome': outcome,
        'outcome_label': [OUTCOME_LABELS[o] for o in outcome]
    })

# ─────────────────────────────────────────────
# ENSURE REQUIRED COLUMNS EXIST
# ─────────────────────────────────────────────
required = ['MAT', 'FC', 'SHTK', 'BF']
missing  = [c for c in required if c not in df.columns]
if missing:
    print(f"\n  ⚠️  Missing columns: {missing}")
    print(f"  Please check your metadata.csv column names")
    print(f"  Expected: MAT, FC, SHTK, BF")
    exit(1)

# Add outcome_label if missing
if 'outcome' in df.columns and 'outcome_label' not in df.columns:
    OUTCOME_LABELS = {0:'Safe', 1:'Risk of Crack', 2:'Crack',
                      3:'Wrinkle', 4:'Wrinkling Tendency'}
    df['outcome_label'] = df['outcome'].map(OUTCOME_LABELS).fillna('Safe')

# Add dummy outputs if not in metadata
if 'max_stress' not in df.columns:
    df['max_stress'] = (200 + 150*(df['BF']/500) - 100*(df['SHTK']/1.0)
                        + 80*df['MAT'] + np.random.normal(0, 12, len(df)))
if 'thinning' not in df.columns:
    df['thinning']   = (0.05 + 0.12*(df['BF']/500)
                        + np.random.normal(0, 0.005, len(df))).clip(0, 0.4)
if 'springback' not in df.columns:
    df['springback'] = (5.0 - 3.0*(df['BF']/500)
                        + np.random.normal(0, 0.3, len(df))).clip(0, 15)
if 'outcome' not in df.columns:
    crack = (df['BF'] > 420) & (df['SHTK'] < 0.965) & (df['MAT'] > 1.05)
    crack_risk = (df['BF'] > 350) & (df['SHTK'] < 0.975) & ~crack
    wrinkle = (df['BF'] < 180) & (df['FC'] > 0.12)
    df['outcome'] = 0
    df.loc[wrinkle, 'outcome'] = 3
    df.loc[crack_risk, 'outcome'] = 1
    df.loc[crack, 'outcome'] = 2
    OUTCOME_LABELS = {0:'Safe', 1:'Risk of Crack', 2:'Crack',
                      3:'Wrinkle', 4:'Wrinkling Tendency'}
    df['outcome_label'] = df['outcome'].map(OUTCOME_LABELS)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
df.to_csv('data/forming_data.csv', index=False)

print(f"\n  Dataset summary:")
print(f"    Rows     : {len(df):,}")
print(f"    Columns  : {df.columns.tolist()}")
print(f"\n  Parameter ranges:")
for p in ['MAT','FC','SHTK','BF']:
    print(f"    {p:<6}: {df[p].min():.4f} → {df[p].max():.4f}")
print(f"\n  Outcome distribution:")
for label, count in df['outcome_label'].value_counts().items():
    print(f"    {label:<22}: {count:>5} ({count/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Dataset Overview — Parameter & Output Distributions',
             fontsize=14, fontweight='bold')

cols   = ['MAT','FC','SHTK','BF','max_stress','thinning','springback','outcome_label']
colors = ['#4C72B0','#55A868','#C44E52','#8172B2',
          '#CCB974','#64B5CD','#E07B39','#9B59B6']

for i, (col, color) in enumerate(zip(cols, colors)):
    ax = axes[i // 4][i % 4]
    if col == 'outcome_label':
        counts = df[col].value_counts()
        ax.bar(range(len(counts)), counts.values,
               color=colors[:len(counts)], alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=7)
        ax.set_title('FLD Outcome Distribution')
    else:
        ax.hist(df[col], bins=40, color=color, alpha=0.85, edgecolor='white')
        ax.set_title(col)
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/1_data_distributions.png', dpi=150, bbox_inches='tight')
print(f"\n  Saved → data/forming_data.csv")
print(f"  Plot  → results/1_data_distributions.png")
print(f"\n✅ Script 1 complete\n")
