"""
Deterministic, per-class holdout evaluation for MedEViT-SAM.

WHY THIS SCRIPT EXISTS (read this before running)
---------------------------------------------------
The original `evaluate_model.py` reuses `MedicalDecathlonDataset`'s
"one-shot class isolation" logic at evaluation time too. That logic calls
`np.random.choice(foreground_labels)` with no fixed seed, so for every
multi-class slice (Liver, Pancreas, Hepatic Vessel, Brain Tumour,
Hippocampus, Prostate) a DIFFERENT random class gets scored every time you
run it. Two consequences:

  1. Results are not reproducible run-to-run for multi-class tasks.
  2. There is no way to recover a separate organ-vs-tumor (DSC1/DSC2 style)
     breakdown, because each slice only ever contributes ONE randomly
     chosen class to the running average.

This script fixes both by explicitly iterating over every foreground class
present in every held-out slice, every time, in the same order. It also
fixes the NSD tolerance so a "2 pixel" tolerance means roughly the same
real distance regardless of how much a given scan had to be upscaled to
reach 512x512 (small native-resolution scans like Brain/Hippocampus were
getting an unfairly strict tolerance under the old fixed-pixel definition).

WHAT THIS SCRIPT GIVES YOU
---------------------------
- Per (task, class) Dice / IoU / NSD           -> rebuilds Table 1's DSC1/DSC2 rows
- Per task "Avg" (mean across that task's classes) -> rebuilds Table 1's "Avg" column
- Overall MACRO (mean of the 10 per-task Avg values)  -> matches your abstract-style number
- Overall MICRO (mean across every individual instance, weighted by how many
  slices/classes each task contributes) -> matches your conclusion-style number

Report ONE of these two "overall" numbers consistently throughout the paper,
and say explicitly which one you're using.

- ALSO writes a per-patient boxplot dataset and PNG (one data point per
  patient per task, matching a patient-level-holdout box plot like the one
  in the manuscript's Figure 5/8). This adds no extra model forward passes
  -- every slice/class score is simply tagged with which patient it came
  from as it's computed, then averaged within each patient afterward.

WHAT ABOUT TRUE PHYSICAL (mm-based) NSD TOLERANCE?
---------------------------------------------------
The precomputed slice cache (from precompute_slice_cache.py) does not store
voxel spacing (mm/pixel), so a fully clinical-standard, per-organ millimeter
tolerance isn't available from the cache alone. This script instead
normalizes the tolerance by each slice's own resize factor (native pixels,
not canvas pixels), which removes the biggest, most obviously wrong source
of distortion. If you want true mm-based tolerances later, you'd need to
extend precompute_slice_cache.py to also store `nii.header.get_zooms()`
per volume, and thread that through here -- happy to add that if you want it.

USAGE
-----
python evaluate_holdout_per_class.py \
    --checkpoint /path/to/best_model_dice.pth \
    --slice_cache_dir /path/to/msd_slice_cache_all10_m150_multiclass \
    --model efficientvit-sam-l0 \
    --image_size 512 \
    --val_ratio 0.2 \
    --split_seed 42 \
    --batch_size 16 \
    --output_json holdout_evaluation_per_class.json \
    --output_csv holdout_evaluation_per_class.csv

--val_ratio and --split_seed MUST match whatever you used during training
(the workflow guide's defaults are 0.2 and 42) or you will not be evaluating
on the exact patients that were actually held out.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
try:
    from data.dataset import (
        MedicalDecathlonDataset,
        get_val_transforms,
        patient_level_holdout_split,
        get_sample_volume_key,
    )
    from models.losses import compute_dice_score, compute_iou_score, compute_nsd_score
except ImportError:
    from medical_dataset import (
        MedicalDecathlonDataset,
        get_val_transforms,
        patient_level_holdout_split,
        get_sample_volume_key,
    )
    from losses import compute_dice_score, compute_iou_score, compute_nsd_score

DEFAULT_TASKS = [
    'Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus',
    'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel',
    'Task09_Spleen', 'Task10_Colon'
]

# Human-readable class names, used only for pretty printing / CSV export.
TASK_LABELS = {
    'Task01_BrainTumour': {1: 'Edema', 2: 'Non-enhancing', 3: 'Enhancing core'},
    'Task02_Heart': {1: 'Left Atrium'},
    'Task03_Liver': {1: 'Liver', 2: 'Tumor'},
    'Task04_Hippocampus': {1: 'Anterior', 2: 'Posterior'},
    'Task05_Prostate': {1: 'Peripheral Zone (PZ)', 2: 'Central Gland (CG)'},
    'Task06_Lung': {1: 'Tumor'},
    'Task07_Pancreas': {1: 'Pancreas', 2: 'Tumor'},
    'Task08_HepaticVessel': {1: 'Vessel', 2: 'Tumor'},
    'Task09_Spleen': {1: 'Spleen'},
    'Task10_Colon': {1: 'Tumor'},
}


def get_bbox_from_mask(mask: np.ndarray, margin: int = 10) -> list:
    """Same convention used everywhere else in the codebase (margin=10)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return [w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10]
    x_min = max(0, xs.min() - margin)
    y_min = max(0, ys.min() - margin)
    x_max = min(mask.shape[1], xs.max() + margin)
    y_max = min(mask.shape[0], ys.max() + margin)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def load_finetuned_model(checkpoint_path, model_name='efficientvit-sam-l0', device='cuda'):
    print(f'Loading model from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_efficientvit_sam_model(name=model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    best_metric = checkpoint.get('best_metric', 'N/A')
    return model.to(device).eval(), epoch, best_metric


def evaluate_full(model, val_samples, transform, image_size, device,
                   batch_size=16, base_tolerance_native_px=2.0):
    """
    Runs every foreground class of every held-out slice through the model,
    exactly once, deterministically.

    Returns TWO dicts:
      - results:         (task_name, class_label) -> list of {'dice','iou','nsd'}
                          (same as before -- rebuilds Table 1)
      - patient_results:  (task_name, volume_key)  -> list of {'dice','iou','nsd'}
                          (NEW -- every slice/class score tagged with which
                          patient it came from, needed for the per-patient
                          boxplot. This adds no extra model forward passes,
                          just bookkeeping.)
    """
    results = defaultdict(list)
    patient_results = defaultdict(list)
    n = len(val_samples)

    for start in tqdm(range(0, n, batch_size), desc='Evaluating (per-class, deterministic)'):
        batch = val_samples[start:start + batch_size]

        imgs_rgb_orig, labels_orig, tasks, orig_shapes, img_tensors, volume_keys = [], [], [], [], [], []

        for s in batch:
            cached = np.load(s['cached_path'], allow_pickle=False)
            img = cached['image'].astype(np.float32, copy=False)
            label = cached['label'].astype(np.float32, copy=False)
            img_rgb = np.stack([img] * 3, axis=-1)

            imgs_rgb_orig.append(img_rgb)
            labels_orig.append(label)
            tasks.append(s['task'])
            orig_shapes.append(img.shape)  # (H_orig, W_orig) -- BEFORE any resize
            # Same patient-identification logic used for the train/val split itself,
            # so "which patient does this slice belong to" is guaranteed consistent
            # with how the holdout set was built in the first place.
            volume_keys.append(get_sample_volume_key(s))

            img_tensors.append(transform(image=img_rgb)['image'])

        img_batch = torch.stack(img_tensors).to(device)

        with torch.no_grad():
            image_embeddings_batch = model.image_encoder(img_batch)

        for i in range(len(batch)):
            task_name = tasks[i]
            label = labels_orig[i]
            img_rgb = imgs_rgb_orig[i]
            orig_h, orig_w = orig_shapes[i]
            volume_key = volume_keys[i]

            # How much this specific slice was scaled to reach `image_size`.
            # A fixed pixel tolerance means a different real distance for every
            # slice unless we account for this.
            resize_scale = image_size / max(orig_h, orig_w)
            effective_tolerance = base_tolerance_native_px * resize_scale

            foreground_labels = np.unique(label)
            foreground_labels = foreground_labels[foreground_labels > 0]
            if len(foreground_labels) == 0:
                continue

            img_embed = image_embeddings_batch[i:i + 1]

            for class_label in foreground_labels:
                class_label = int(class_label)
                isolated = (label == class_label).astype(np.float32)

                transformed = transform(image=img_rgb, mask=isolated)
                t_mask = transformed['mask']
                if t_mask.ndim == 2:
                    t_mask = t_mask.unsqueeze(0)
                mask_np = t_mask[0].numpy()

                bbox = get_bbox_from_mask(mask_np)
                bbox_tensor = torch.tensor([bbox], dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    sparse, dense = model.prompt_encoder(points=None, boxes=bbox_tensor, masks=None)
                    pred_mask, _ = model.mask_decoder(
                        image_embeddings=img_embed,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                    )
                    if pred_mask.shape[-2:] != t_mask.shape[-2:]:
                        pred_mask = F.interpolate(
                            pred_mask.float(), size=t_mask.shape[-2:],
                            mode='bilinear', align_corners=False,
                        )

                target = t_mask.unsqueeze(0).float().to(device)

                dice = compute_dice_score(pred_mask, target)
                iou = compute_iou_score(pred_mask, target)
                nsd = compute_nsd_score(pred_mask, target, tolerance=effective_tolerance)

                score = {'dice': dice, 'iou': iou, 'nsd': nsd}
                results[(task_name, class_label)].append(score)
                patient_results[(task_name, volume_key)].append(score)

    return results, patient_results


def aggregate_results(results):
    per_class = {}
    per_task_class_aggs = defaultdict(list)

    for (task, class_label), items in results.items():
        agg = {
            'class_name': TASK_LABELS.get(task, {}).get(class_label, f'Class {class_label}'),
            'n': len(items),
            'dice': float(np.mean([x['dice'] for x in items])),
            'iou': float(np.mean([x['iou'] for x in items])),
            'nsd': float(np.mean([x['nsd'] for x in items])),
        }
        per_class.setdefault(task, {})[str(class_label)] = agg
        per_task_class_aggs[task].append(agg)

    per_task = {}
    for task, class_aggs in per_task_class_aggs.items():
        per_task[task] = {
            'dice_avg': float(np.mean([c['dice'] for c in class_aggs])),
            'iou_avg': float(np.mean([c['iou'] for c in class_aggs])),
            'nsd_avg': float(np.mean([c['nsd'] for c in class_aggs])),
            'n_total': int(sum(c['n'] for c in class_aggs)),
            'n_classes': len(class_aggs),
        }

    macro_dice = float(np.mean([v['dice_avg'] for v in per_task.values()]))
    macro_iou = float(np.mean([v['iou_avg'] for v in per_task.values()]))
    macro_nsd = float(np.mean([v['nsd_avg'] for v in per_task.values()]))

    all_dice, all_iou, all_nsd = [], [], []
    for items in results.values():
        all_dice.extend(x['dice'] for x in items)
        all_iou.extend(x['iou'] for x in items)
        all_nsd.extend(x['nsd'] for x in items)

    micro_dice = float(np.mean(all_dice))
    micro_iou = float(np.mean(all_iou))
    micro_nsd = float(np.mean(all_nsd))

    return {
        'per_class': per_class,
        'per_task': per_task,
        'overall': {
            'macro_dice': macro_dice, 'macro_iou': macro_iou, 'macro_nsd': macro_nsd,
            'micro_dice': micro_dice, 'micro_iou': micro_iou, 'micro_nsd': micro_nsd,
            'n_instances_total': len(all_dice),
            'definition_note': (
                'macro_* = unweighted mean across the 10 tasks Avg column '
                '(matches an abstract-style "overall" number). '
                'micro_* = mean across every individual (slice, class) instance, '
                'weighted by how much data each task contributes '
                '(matches a conclusion/holdout-log-style "overall" number). '
                'Use exactly ONE of these consistently in the paper.'
            ),
        },
    }


def aggregate_per_patient(patient_results):
    """
    Collapses every slice/class score down to ONE (dice, iou, nsd) value
    per patient per task (averaged across whatever slices and classes
    belonged to that patient for that task), then returns, for each task,
    the list of per-patient values -- exactly what a boxplot needs: one
    data point per patient, not per slice.
    """
    per_task_per_patient = defaultdict(dict)  # task -> {volume_key: {dice, iou, nsd, n}}

    for (task, volume_key), items in patient_results.items():
        per_task_per_patient[task][volume_key] = {
            'dice': float(np.mean([x['dice'] for x in items])),
            'iou': float(np.mean([x['iou'] for x in items])),
            'nsd': float(np.mean([x['nsd'] for x in items])),
            'n_slices_classes': len(items),
        }

    boxplot_data = {}
    sanity_warnings = []
    for task, patients in per_task_per_patient.items():
        avg_items_per_patient = float(np.mean([p['n_slices_classes'] for p in patients.values()]))
        if avg_items_per_patient < 2.0:
            sanity_warnings.append(
                f"  WARNING: {task} averages only {avg_items_per_patient:.1f} scored "
                f"items per detected 'patient' -- patient grouping may be broken "
                f"(every slice being treated as its own patient). Expected roughly "
                f"10-150 for this dataset."
            )
        boxplot_data[task] = {
            'dice': [p['dice'] for p in patients.values()],
            'iou': [p['iou'] for p in patients.values()],
            'nsd': [p['nsd'] for p in patients.values()],
            'n_patients': len(patients),
            'avg_items_per_patient': avg_items_per_patient,
        }

    if sanity_warnings:
        print('\n' + '!' * 100)
        print('PATIENT-GROUPING SANITY CHECK FAILED for one or more tasks:')
        for w in sanity_warnings:
            print(w)
        print('!' * 100 + '\n')
    else:
        print('\nPatient-grouping sanity check passed for all tasks '
              '(each detected patient has a sensible number of scored items).\n')

    return per_task_per_patient, boxplot_data


def plot_boxplots(boxplot_data, output_path, tasks_order):
    """
    Draws one figure with three panels (Dice, IoU, NSD), each showing one
    box per task, where every point inside a box is one patient's mean
    score on that task -- matching a patient-level-holdout box plot like
    the one described in the manuscript (Fig. 5 / Fig. 8).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['dice', 'iou', 'nsd']
    metric_titles = {'dice': 'Dice Similarity Coefficient (DSC)', 'iou': 'Intersection over Union (IoU)', 'nsd': 'Normalized Surface Distance (NSD)'}

    available_tasks = [t for t in tasks_order if t in boxplot_data and len(boxplot_data[t]['dice']) > 0]
    short_labels = [t.split('_', 1)[1] if '_' in t else t for t in available_tasks]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, metric in zip(axes, metrics):
        data = [boxplot_data[t][metric] for t in available_tasks]
        ax.boxplot(data, labels=short_labels, showmeans=True)
        ax.set_title(metric_titles[metric], fontsize=12)
        ax.set_ylabel(metric_titles[metric])
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Boxplot figure saved to: {output_path}')


def print_table(aggregated):
    print('\n' + '=' * 100)
    print(f"{'Task':<24}{'Class':<20}{'N':>7}{'Dice':>9}{'IoU':>9}{'NSD':>9}")
    print('-' * 100)
    for task in DEFAULT_TASKS:
        if task not in aggregated['per_class']:
            continue
        classes = aggregated['per_class'][task]
        for class_id in sorted(classes.keys(), key=int):
            c = classes[class_id]
            print(f"{task:<24}{c['class_name']:<20}{c['n']:>7}{c['dice']*100:>8.2f}%{c['iou']*100:>8.2f}%{c['nsd']*100:>8.2f}%")
        t = aggregated['per_task'][task]
        print(f"{'':<24}{'-- task Avg --':<20}{t['n_total']:>7}{t['dice_avg']*100:>8.2f}%{t['iou_avg']*100:>8.2f}%{t['nsd_avg']*100:>8.2f}%")
        print('-' * 100)

    o = aggregated['overall']
    print(f"\nOVERALL (MACRO -- unweighted mean over 10 tasks):  Dice {o['macro_dice']*100:.2f}%  IoU {o['macro_iou']*100:.2f}%  NSD {o['macro_nsd']*100:.2f}%")
    print(f"OVERALL (MICRO -- mean over all {o['n_instances_total']} instances): Dice {o['micro_dice']*100:.2f}%  IoU {o['micro_iou']*100:.2f}%  NSD {o['micro_nsd']*100:.2f}%")
    print('=' * 100 + '\n')


def write_csv(aggregated, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'class_id', 'class_name', 'n', 'dice', 'iou', 'nsd'])
        for task, classes in aggregated['per_class'].items():
            for class_id, c in classes.items():
                writer.writerow([task, class_id, c['class_name'], c['n'], c['dice'], c['iou'], c['nsd']])
        for task, t in aggregated['per_task'].items():
            writer.writerow([task, 'AVG', 'task_avg', t['n_total'], t['dice_avg'], t['iou_avg'], t['nsd_avg']])
        o = aggregated['overall']
        writer.writerow(['OVERALL', 'MACRO', 'macro_avg', '', o['macro_dice'], o['macro_iou'], o['macro_nsd']])
        writer.writerow(['OVERALL', 'MICRO', 'micro_avg', o['n_instances_total'], o['micro_dice'], o['micro_iou'], o['micro_nsd']])
    print(f'CSV written to: {path}')


def main():
    parser = argparse.ArgumentParser(description='Deterministic, per-class holdout evaluation for MedEViT-SAM')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--slice_cache_dir', type=str, required=True,
                         help='Directory containing slice_index.json + precomputed .npz slices')
    parser.add_argument('--base_dir', type=str, default='.',
                         help='Unused when --slice_cache_dir resolves successfully; kept for API compatibility')
    parser.add_argument('--model', type=str, default='efficientvit-sam-l0')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                         help='MUST match the value used during training')
    parser.add_argument('--split_seed', type=int, default=42,
                         help='MUST match the value used during training')
    parser.add_argument('--nsd_tolerance_native_px', type=float, default=2.0,
                         help='Tolerance in *native* (pre-resize) pixels, scaled per-slice by its own resize factor')
    parser.add_argument('--output_json', type=str, default='holdout_evaluation_per_class.json')
    parser.add_argument('--output_csv', type=str, default='holdout_evaluation_per_class.csv')
    parser.add_argument('--output_boxplot_json', type=str, default='holdout_boxplot_data_per_patient.json',
                         help='Per-patient values used to draw the boxplot (one number per patient per task)')
    parser.add_argument('--output_boxplot_png', type=str, default='holdout_boxplot_per_patient.png')
    parser.add_argument('--skip_boxplot', action='store_true',
                         help='Skip generating the boxplot PNG (still saves the per-patient JSON)')
    parser.add_argument('--seed', type=int, default=42, help='Global seed (no longer used for class choice, kept for hygiene)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = MedicalDecathlonDataset(
        base_dir=args.base_dir,
        tasks=DEFAULT_TASKS,
        split='training',
        transform=None,
        balance_tasks=False,
        cached_slice_dir=args.slice_cache_dir,
    )

    _, val_samples, info = patient_level_holdout_split(
        dataset.samples, val_ratio=args.val_ratio, seed=args.split_seed
    )
    print(f"Holdout: {info['n_val_slices']} slices from {info['n_val_volumes']} volumes "
          f"(train had {info['n_train_volumes']} volumes)")

    model, epoch, best_metric = load_finetuned_model(args.checkpoint, args.model, device)
    print(f'Model epoch: {epoch} | training best metric: {best_metric}')

    transform = get_val_transforms(args.image_size)

    results, patient_results = evaluate_full(
        model, val_samples, transform, args.image_size, device,
        batch_size=args.batch_size,
        base_tolerance_native_px=args.nsd_tolerance_native_px,
    )

    aggregated = aggregate_results(results)
    print_table(aggregated)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'model': args.model,
                'image_size': args.image_size,
                'val_ratio': args.val_ratio,
                'split_seed': args.split_seed,
                'nsd_tolerance_native_px': args.nsd_tolerance_native_px,
                'n_val_volumes': info['n_val_volumes'],
                'n_val_slices': info['n_val_slices'],
            },
            **aggregated,
        }, f, indent=2)
    print(f'JSON written to: {args.output_json}')

    write_csv(aggregated, args.output_csv)

    # --- Per-patient boxplot data (one value per patient per task) ---
    per_task_per_patient, boxplot_data = aggregate_per_patient(patient_results)

    Path(args.output_boxplot_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_boxplot_json, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'val_ratio': args.val_ratio,
                'split_seed': args.split_seed,
                'note': (
                    'Each value in dice/iou/nsd lists is ONE PATIENT\'s mean score '
                    '(averaged across all their held-out slices and foreground classes '
                    'for that task) -- ready to feed directly into a boxplot.'
                ),
            },
            'boxplot_data': boxplot_data,
        }, f, indent=2)
    print(f'Per-patient boxplot data written to: {args.output_boxplot_json}')

    if not args.skip_boxplot:
        try:
            plot_boxplots(boxplot_data, args.output_boxplot_png, DEFAULT_TASKS)
        except Exception as e:
            print(f'Could not generate boxplot PNG ({e}). '
                  f'The raw per-patient data is still in {args.output_boxplot_json} -- '
                  f'you can plot it yourself from there.')


if __name__ == '__main__':
    main()