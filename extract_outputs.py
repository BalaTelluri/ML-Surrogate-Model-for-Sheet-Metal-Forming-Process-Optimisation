"""
Extract Real Outputs from DDACS HDF5 Files
============================================
Extracts real FEM simulation outputs from HDF5 files:
- Max Von Mises stress (from element_shell_stress)
- Thinning (from element_shell_thickness)
- Springback (from OP20 displacement)
- FLD outcome classification

Merges with metadata.csv to create complete dataset.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
H5_DIR       = Path('/Users/bala/ddacs_data/h5')
METADATA_PATH= Path('/Users/bala/ddacs_data/metadata.csv')
OUTPUT_PATH  = Path('data/metadata_with_outputs.csv')

h5_files = list(H5_DIR.glob('*.h5'))
print(f"Found {len(h5_files)} HDF5 files")

# Load metadata
meta = pd.read_csv(METADATA_PATH)
# Normalise BF if in Newtons
if meta['BF'].max() > 1000:
    meta['BF'] = meta['BF'] / 1000
print(f"Metadata: {len(meta)} rows")
print(f"Columns: {meta.columns.tolist()}")

# ─────────────────────────────────────────────
# EXTRACTION FUNCTION
# ─────────────────────────────────────────────
def extract_outputs(h5_path):
    """
    Extract key outputs from one HDF5 simulation file.
    Returns dict with sim_id and output values.
    """
    sim_id = int(h5_path.stem)

    with h5py.File(h5_path, 'r') as f:

        # ── Max Von Mises Stress (MPa) ──
        # element_shell_stress shape: (timesteps, elements, components)
        # Von Mises stress = last timestep, all elements
        stress_data = np.array(f['OP10/blank/element_shell_stress'])
        # Last timestep, all elements — compute Von Mises from stress components
        # stress components: [s11, s22, s33, s12, s23, s13] or similar
        if stress_data.ndim == 3:
            s = stress_data[-1]  # last timestep
        else:
            s = stress_data

        if s.shape[-1] >= 3:
            # Von Mises from principal stresses
            s11, s22, s12 = s[:, 0], s[:, 1], s[:, 3] if s.shape[-1] > 3 else np.zeros(len(s))
            von_mises = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
        else:
            von_mises = np.abs(s).max(axis=-1)

        max_stress = float(np.nanmax(von_mises))

        # ── Thinning ──
        # thickness shape: (timesteps, elements) or (elements,)
        thickness_data = np.array(f['OP10/blank/element_shell_thickness'])
        if thickness_data.ndim == 2:
            initial_thickness = thickness_data[0]   # first timestep
            final_thickness   = thickness_data[-1]  # last timestep
        else:
            initial_thickness = thickness_data
            final_thickness   = thickness_data

        # Thinning = (initial - final) / initial
        with np.errstate(divide='ignore', invalid='ignore'):
            thinning_per_element = (initial_thickness - final_thickness) / initial_thickness
        thinning = float(np.nanmax(thinning_per_element))  # worst case thinning

        # ── Springback (OP20) ──
        # Displacement after springback operation
        try:
            disp_op20 = np.array(f['OP20/blank/node_displacement'])
            if disp_op20.ndim == 3:
                # shape: (timesteps, nodes, xyz)
                z_disp = disp_op20[-1, :, 2]  # last timestep, z-direction
            else:
                z_disp = disp_op20[:, 2]
            springback = float(np.nanmax(np.abs(z_disp)))
        except:
            springback = 0.0

        # ── Effective Plastic Strain ──
        strain_data = np.array(
            f['OP10/blank/element_shell_effective_plastic_strain']
        )
        if strain_data.ndim == 2:
            max_strain = float(np.nanmax(strain_data[-1]))
        else:
            max_strain = float(np.nanmax(strain_data))

    return {
        'ID':         sim_id,
        'max_stress': max_stress,
        'thinning':   thinning,
        'springback': springback,
        'max_strain': max_strain
    }

# ─────────────────────────────────────────────
# CLASSIFY FLD OUTCOME FROM REAL DATA
# ─────────────────────────────────────────────
def classify_outcome(thinning, max_strain):
    """
    FLD classification based on thinning and strain thresholds
    from Heinzelmann et al. 2025
    """
    if thinning > 0.25 or max_strain > 0.4:
        return 2, 'Crack'
    elif thinning > 0.15 or max_strain > 0.25:
        return 1, 'Risk of Crack'
    elif thinning < 0.02 and max_strain < 0.05:
        return 3, 'Wrinkle'
    else:
        return 0, 'Safe'

# ─────────────────────────────────────────────
# EXTRACT ALL FILES
# ─────────────────────────────────────────────
records  = []
errors   = []
n_files  = len(h5_files)

print(f"\nExtracting outputs from {n_files} files...")
print(f"{'Progress':>10} | {'File':>12} | {'Max Stress':>12} | {'Thinning':>10} | Outcome")
print(f"{'-'*65}")

for i, h5_path in enumerate(h5_files):
    try:
        result = extract_outputs(h5_path)
        outcome_code, outcome_label = classify_outcome(
            result['thinning'], result['max_strain']
        )
        result['outcome']       = outcome_code
        result['outcome_label'] = outcome_label
        records.append(result)

        if (i + 1) % 100 == 0 or i < 5:
            print(f"  {i+1:>6}/{n_files} | {h5_path.stem:>12} | "
                  f"{result['max_stress']:>10.1f} | "
                  f"{result['thinning']:>10.4f} | {outcome_label}")

    except Exception as e:
        errors.append({'file': h5_path.name, 'error': str(e)})

print(f"\nExtracted: {len(records)} files")
print(f"Errors:    {len(errors)} files")

# ─────────────────────────────────────────────
# MERGE WITH METADATA
# ─────────────────────────────────────────────
outputs_df = pd.DataFrame(records)
print(f"\nOutputs shape: {outputs_df.shape}")

# Merge on ID
merged = meta.merge(outputs_df, on='ID', how='inner')
print(f"Merged shape: {merged.shape}")
print(f"\nOutcome distribution:")
for label, count in merged['outcome_label'].value_counts().items():
    print(f"  {label:<22}: {count:>5} ({count/len(merged)*100:.1f}%)")

print(f"\nOutput ranges:")
print(f"  max_stress: {merged['max_stress'].min():.1f} → {merged['max_stress'].max():.1f} MPa")
print(f"  thinning  : {merged['thinning'].min():.4f} → {merged['thinning'].max():.4f}")
print(f"  springback: {merged['springback'].min():.4f} → {merged['springback'].max():.4f}")

# Save
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved → {OUTPUT_PATH}")
print(f"\nNow copy this to your project:")
print(f"  cp {OUTPUT_PATH} ~/Downloads/final_project/data/metadata.csv")
print(f"\nThen run: bash run_pipeline.sh")
