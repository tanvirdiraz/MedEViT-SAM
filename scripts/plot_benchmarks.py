import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for p in [str(_repo_dir), str(_repo_dir / 'data'), str(_repo_dir / 'models'), str(_repo_dir / 'configs'), str(_repo_dir / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set writable matplotlib cache dir if needed
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "matplotlib"))

# Output directory: saves directly to the project's figures directory
output_dir = Path(__file__).resolve().parent.parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Evaluation Data (7 models - NVDLMED & Kim et al. removed)
# Numbers matching Table I (DSC) and Table II (NSD) in manuscript
# ---------------------------------------------------------
models = [
    "CerebriuDIKU",
    "nnUNet",
    "C2FNAS",
    "DiNTS",
    "SWIN UNETR",
    "Universal",
    "MedEViT-SAM"
]

dsc_scores = [67.01, 77.89, 76.97, 77.93, 78.67, 79.87, 85.09]
nsd_scores = [77.86, 88.09, 87.83, 88.68, 89.28, 90.34, 82.13]

# Set global figure style
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

# ---------------------------------------------------------
# Plot 1: Overall DSC (Vertical Bar Chart - Ranked Ascending)
# ---------------------------------------------------------
sorted_dsc = sorted(zip(models, dsc_scores), key=lambda x: x[1])
dsc_models, dsc_vals = zip(*sorted_dsc)

# Colors matching viridis palette
cmap_viridis = plt.cm.viridis(np.linspace(0.15, 0.90, len(dsc_models)))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars = ax.bar(dsc_models, dsc_vals, color=cmap_viridis, width=0.8)

# Customizing labels and limits with original large, bold fonts
ax.set_xlabel("Metric", fontsize=30, fontweight='bold', labelpad=14)
ax.set_ylabel("Overall DSC", fontsize=30, fontweight='bold', labelpad=14)
ax.set_ylim(0, 100)

# Tick labels size and styling (matching original large size)
plt.xticks(rotation=45, ha="right", fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')

plt.tight_layout()

# Save to both figure_3.png (manuscript) and overall_dsc_vertical.png
dsc_save_path = output_dir / "figure_3.png"
plt.savefig(dsc_save_path, dpi=300)
plt.savefig(output_dir / "overall_dsc_vertical.png", dpi=300)
plt.close()
print(f"Overall DSC plot saved to: {dsc_save_path}")

# ---------------------------------------------------------
# Plot 2: Overall NSD (Vertical Bar Chart - Ranked Ascending)
# ---------------------------------------------------------
sorted_nsd = sorted(zip(models, nsd_scores), key=lambda x: x[1])
nsd_models, nsd_vals = zip(*sorted_nsd)

# Colors matching magma palette
cmap_magma = plt.cm.magma(np.linspace(0.15, 0.88, len(nsd_models)))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars = ax.bar(nsd_models, nsd_vals, color=cmap_magma, width=0.8)

# Customizing labels and limits with original large, bold fonts
ax.set_xlabel("Metric", fontsize=30, fontweight='bold', labelpad=14)
ax.set_ylabel("Overall NSD", fontsize=30, fontweight='bold', labelpad=14)
ax.set_ylim(0, 100)

# Tick labels size and styling (matching original large size)
plt.xticks(rotation=45, ha="right", fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')

plt.tight_layout()

# Save to both figure_4.png (manuscript) and overall_nsd_vertical.png
nsd_save_path = output_dir / "figure_4.png"
plt.savefig(nsd_save_path, dpi=300)
plt.savefig(output_dir / "overall_nsd_vertical.png", dpi=300)
plt.close()
print(f"Overall NSD plot saved to: {nsd_save_path}")

print("Successfully regenerated both comparison bar plots with original large bold fonts!")
