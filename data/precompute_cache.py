"""
Precompute 2D Slice Cache from 3D Medical Segmentation Decathlon (MSD) Volumes.

Extracts axial 2D slices from 3D NIfTI volumes, applies aspect-ratio-preserving
resizing to 512x512, and saves them as compressed .npz files for high-throughput
training and deterministic evaluation.
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm


DEFAULT_TASKS = [
    'Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus',
    'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel',
    'Task09_Spleen', 'Task10_Colon'
]


def resize_and_pad(image_slice, label_slice, target_size=512):
    """
    Resizes the slice preserving its native aspect ratio so that the longest dimension
    is target_size, then symmetrically zero-pads to (target_size, target_size).
    """
    h, w = image_slice.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # Resize image with bilinear interpolation, label with nearest neighbor
    resized_img = cv2.resize(image_slice, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_lbl = cv2.resize(label_slice.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Compute symmetric padding
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded_img = np.pad(resized_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    padded_lbl = np.pad(resized_lbl, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_img, padded_lbl, scale


def normalize_intensity(img_data):
    """Clips intensities to [2nd, 98th] percentiles and normalizes to [0, 1]."""
    p2, p98 = np.percentile(img_data, 2), np.percentile(img_data, 98)
    if p98 > p2:
        img_data = np.clip(img_data, p2, p98)
        img_data = (img_data - p2) / (p98 - p2)
    else:
        img_data = np.zeros_like(img_data)
    return img_data.astype(np.float32)


def process_task(task_dir: Path, out_dir: Path, image_size: int = 512, max_slices: int = 150):
    task_name = task_dir.name
    dataset_json = task_dir / "dataset.json"
    if not dataset_json.exists():
        print(f"Warning: {dataset_json} not found. Skipping {task_name}.")
        return []

    with open(dataset_json, 'r') as f:
        meta = json.load(f)

    training_samples = meta.get('training', [])
    records = []

    print(f"\nProcessing {task_name} ({len(training_samples)} volumes)...")
    for sample in tqdm(training_samples, desc=task_name):
        img_rel = sample['image'].lstrip('./')
        lbl_rel = sample['label'].lstrip('./')
        img_path = task_dir / img_rel
        lbl_path = task_dir / lbl_rel

        if not img_path.exists() or not lbl_path.exists():
            continue

        volume_id = Path(img_rel).name.replace('.nii.gz', '').replace('.nii', '')

        # Load NIfTI volumes
        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))
        img_data = img_nii.get_fdata(dtype=np.float32)
        lbl_data = lbl_nii.get_fdata(dtype=np.float32)

        # Handle 4D volumes (e.g. Brain Tumour has 4 modalities) -> take first or average
        if img_data.ndim == 4:
            img_data = img_data[..., 0]

        img_data = normalize_intensity(img_data)
        num_slices = img_data.shape[2]

        for z in range(num_slices):
            lbl_slice = lbl_data[:, :, z]
            # Keep all slices that have foreground objects
            has_fg = np.any(lbl_slice > 0)
            if not has_fg and (z % 5 != 0):  # Subsample background slices to prevent imbalance
                continue

            img_slice = img_data[:, :, z]
            proc_img, proc_lbl, scale = resize_and_pad(img_slice, lbl_slice, target_size=image_size)

            filename = f"{task_name}__{volume_id}__z{z:03d}.npz"
            save_path = out_dir / filename
            np.savez_compressed(save_path, image=proc_img, label=proc_lbl)

            records.append({
                'task': task_name,
                'volume_id': volume_id,
                'slice_idx': z,
                'cached_path': str(filename),
                'scale': scale,
                'has_foreground': bool(has_fg)
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Precompute 2D slice cache for MSD tasks")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to unzipped MSD dataset directory")
    parser.add_argument("--out_dir", type=str, default="./msd_slice_cache", help="Output directory for cached .npz files")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS), help="Comma-separated list of tasks to process")
    parser.add_argument("--image_size", type=int, default=512, help="Target image size (default: 512)")
    parser.add_argument("--max_slices_per_volume", type=int, default=150, help="Max slices per volume")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_records = []

    for task_name in task_list:
        task_dir = data_dir / task_name
        if not task_dir.exists():
            print(f"Task directory {task_dir} not found. Skipping.")
            continue
        records = process_task(task_dir, out_dir, image_size=args.image_size, max_slices=args.max_slices_per_volume)
        all_records.extend(records)

    summary_file = out_dir / "cache_index.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)

    print(f"\n[Done] Successfully cached {len(all_records)} slices into {out_dir}")
    print(f"Index summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
