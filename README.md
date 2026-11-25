# MedEViT-SAM: A Unified Foundation Model for Multi-Task Medical Image Segmentation

Official PyTorch implementation for **MedEViT-SAM**, a lightweight prompt-conditioned foundation model designed for unified medical image segmentation across heterogeneous 2D/3D CT and MRI modalities without task-specific heads.

---

## Overview & Key Contributions

Adapting the Segment Anything Model (SAM) to medical imaging is typically hindered by excessive computational complexity (~3,000 GMACs for ViT-H) and a class-agnostic mask decoding formulation that cannot natively differentiate multi-class anatomical structures. 

**MedEViT-SAM addresses both challenges:**
* **Unified Multi-Task Model:** Evaluated across all 10 tasks of the **Medical Segmentation Decathlon (MSD)** using a single set of network weights.
* **Stochastic Class Isolation:** Converts multi-class clinical annotations into prompt-conditioned single-mask targets during training, enabling the class-agnostic SAM decoder to localize arbitrary anatomical targets.
* **85-fold Encoder Complexity Reduction:** Operates at **34.87 GMACs** with an end-to-end latency of **8.73 ms/slice (>114 FPS)** on an NVIDIA RTX 5070 Ti.
* **Superior Overlap:** Achieves **85.09% Overall Macro DSC**, outperforming specialized automated 3D volumetric baselines (Universal Model, SWIN UNETR, nnU-Net).

---

## Benchmark Results

Evaluated across patient-level holdout partitions (80/20 split) on all 10 MSD tasks:

| Method | Type | Overall DSC (%) ↑ | Overall NSD (%) ↑ | Params | Latency / FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CerebriuDIKU | 3D Automated | 67.01 | 77.86 | — | — |
| C2FNAS | 3D Automated | 76.97 | 87.83 | — | — |
| nnU-Net | 3D Automated | 77.89 | 88.09 | — | — |
| DiNTS | 3D Automated | 77.93 | 88.68 | — | — |
| SWIN UNETR | 3D Automated | 78.67 | 89.28 | ~62M | — |
| Universal Model | 3D Automated | 79.87 | **90.34** | ~60M | — |
| **MedEViT-SAM (Ours)** | **Promptable 2D** | **85.09** | 82.13 | **34.79M** | **8.73 ms / 114.5 FPS** |

*Note: Automated 3D baselines output pre-fixed label volumes directly from 3D inputs, whereas MedEViT-SAM evaluates 2D slices with loose bounding-box prompts (±0–10 px jitter).*

---

## Repository Structure

```
Github_medEvit/
├── configs/
│   └── train_config.py         # Hyperparameters, paths, loss weights, and task catalog
├── data/
│   ├── dataset.py              # Multi-task dataset loader & Stochastic Class Isolation
│   └── precompute_cache.py     # Precompute 2D axial slice cache from 3D NIfTI volumes
├── efficientvit/               # EfficientViT backbone and SAM decoder architecture
├── models/
│   └── losses.py               # Compound loss (Tversky + Focal + Boundary) and metrics
├── scripts/
│   ├── ablation_isolation.py   # Reproduce Table III (Isolation vs. Multi-class)
│   ├── ablation_robustness.py  # Reproduce Figs. 9 & 10 (Jitter & NSD tolerance sweeps)
│   ├── measure_efficiency.py   # Reproduce Section V.C (GMACs, latency, parameter count)
│   └── plot_benchmarks.py      # Reproduce Figs. 3 & 4 (Ranked bar charts)
├── train.py                    # Unified 10-task training script
├── evaluate.py                 # Holdout patient-level evaluation (reproduces Tables I & II)
├── predict.py                  # Single-image promptable inference demo
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 License
└── README.md
```

---

## Installation

### Prerequisites
* Linux or Windows
* Python >= 3.10
* NVIDIA GPU with CUDA support (tested on RTX 4090 / RTX 5070 Ti)
* PyTorch >= 2.0.0

### Setup Environment
```bash
git clone https://github.com/tanvirdiraz/MedEViT-SAM.git
cd MedEViT-SAM

conda create -n medevit python=3.10 -y
conda activate medevit

pip install -r requirements.txt
```

---

## Quickstart

### 1. Precompute Slice Cache (Recommended)
To accelerate training across 10 MSD tasks, pre-extract 2D axial slices from raw NIfTI volumes into compressed numpy archives:
```bash
python data/precompute_cache.py \
    --data_dir ./data/msd \
    --out_dir ./msd_slice_cache \
    --image_size 512
```

### 2. Multi-Task Training
Train a unified MedEViT-SAM model across all 10 tasks using stochastic class isolation:
```bash
python train.py \
    --slice_cache_dir ./msd_slice_cache \
    --image_size 512 \
    --batch_size 16 \
    --epochs 35 \
    --lr 5e-5 \
    --min_lr 1e-6 \
    --save_dir ./checkpoints/medevitsam_unified
```

### 3. Quantitative Evaluation (Tables I & II)
Evaluate your trained checkpoint on the deterministic patient-level holdout validation set:
```bash
python evaluate.py \
    --checkpoint ./checkpoints/medevitsam_unified/best_model_dice.pth \
    --slice_cache_dir ./msd_slice_cache \
    --image_size 512 \
    --val_ratio 0.2 \
    --split_seed 42 \
    --batch_size 16 \
    --output_json holdout_evaluation.json \
    --output_csv holdout_evaluation.csv
```

### 4. Interactive Single-Slice Prediction
Run interactive promptable inference on any 2D medical slice with a bounding box prompt:
```bash
python predict.py \
    --image ./sample_slice.png \
    --box "120,150,380,410" \
    --checkpoint ./checkpoints/medevitsam_unified/best_model_dice.pth \
    --output prediction_overlay.png
```

---

## Reproducing Paper Analyses & Ablations

All ablation studies and efficiency benchmarks reported in the paper can be reproduced with a single command:

* **Stochastic Class Isolation Ablation (Table III):**
  ```bash
  python scripts/ablation_isolation.py \
      --checkpoint ./checkpoints/medevitsam_unified/best_model_dice.pth \
      --slice_cache_dir ./msd_slice_cache
  ```

* **Box-Prompt Jitter & NSD Tolerance Sweep (Figures 9 & 10):**
  ```bash
  python scripts/ablation_robustness.py \
      --checkpoint ./checkpoints/medevitsam_unified/best_model_dice.pth \
      --slice_cache_dir ./msd_slice_cache
  ```

* **Complexity, GMACs & GPU Latency Profile (Section V.C):**
  ```bash
  python scripts/measure_efficiency.py
  ```

* **Ranked Benchmark Bar Plots (Figures 3 & 4):**
  ```bash
  python scripts/plot_benchmarks.py
  ```

---

## Pretrained Weights

Pretrained model weights (`best_model_dice.pth`, 139 MB) are available at:
* **Hugging Face:** `[Insert Hugging Face Repo Link]`
* **Zenodo / Google Drive:** `[Insert Download Link]`

Place the downloaded weights file in `./checkpoints/medevitsam_unified/best_model_dice.pth`.

---

## Citation

*This manuscript is currently under review. Complete citation details and DOI will be updated upon publication.*

---

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
