import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for p in [str(_repo_dir), str(_repo_dir / 'data'), str(_repo_dir / 'models'), str(_repo_dir / 'configs'), str(_repo_dir / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

"""
Ablation Study: Stochastic Class Isolation vs. Unisolated (Merged-Prompt) Baseline
==================================================================================
Evaluates on your ALREADY-TRAINED checkpoint without retraining.
Operates on the identical deterministic patient-level holdout subsample
(or full holdout) used for all other evaluations in this project.

RESEARCH QUESTION & PURPOSE
---------------------------
Supervisor Item 12 asks for an ablation demonstrating the effect of the proposed
"Stochastic Class Isolation" strategy compared to an unisolated alternative.

In MedEViT-SAM, the mask decoder is class-agnostic and produces a single binary
mask (multimask_output=False). To enable multi-organ and multi-structure segmentation
across heterogeneous datasets, the model relies on spatial prompt conditioning:
at training time, stochastic class isolation supervises the model on one isolated
foreground target per step with a class-specific bounding box.

This ablation isolates the necessity of that mechanism by comparing:
  (1) PROPOSED (Class-Specific Isolated Prompt):
      Each foreground class on a slice is queried with its own tight bounding box.
      This corresponds to the model's standard evaluation protocol enabled by
      stochastic class isolation.

  (2) ABLATION (No Isolation / Merged Foreground Prompt):
      All foreground classes on the slice are merged into a single union mask,
      and a single enclosing bounding box is provided. The model must attempt
      sub-structure segmentation without class-isolated spatial cues.
      Metrics are evaluated against each constituent class ground-truth to measure
      the loss of anatomical specificity / sub-structure disambiguation.

  (3) NEGATIVE CONTROL (Single-Class vs. Multi-Class Tasks):
      Single-class tasks (Heart, Lung, Spleen, Colon) have only one foreground class,
      so merged box == isolated box. They serve as an internal control (delta = 0%).
      Multi-class tasks (Brain Tumour, Liver, Hippocampus, Prostate, Pancreas, Hepatic Vessel)
      demonstrate severe performance degradation when spatial isolation is removed.

OUTPUTS
-------
  - class_isolation_ablation_summary.json : complete numerical records
  - class_isolation_ablation_summary.csv  : tabular metrics per task and group
  - class_isolation_ablation_barplot.png  : publication-quality comparison figure
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Ensure codes directory and project root are always in sys.path
_codes_dir = Path(__file__).resolve().parent
_project_dir = _codes_dir.parent
for p in [str(_codes_dir), str(_project_dir), str(_codes_dir / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
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
    'Task09_Spleen', 'Task10_Colon',
]

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

MULTI_CLASS_TASKS = [
    'Task01_BrainTumour', 'Task03_Liver', 'Task04_Hippocampus',
    'Task05_Prostate', 'Task07_Pancreas', 'Task08_HepaticVessel'
]

SINGLE_CLASS_TASKS = [
    'Task02_Heart', 'Task06_Lung', 'Task09_Spleen', 'Task10_Colon'
]


def get_bbox_from_mask(mask: np.ndarray, margin: int = 10) -> list:
    """Exact convention used throughout the codebase (margin=10)."""
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
    print(f'Loading model checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_efficientvit_sam_model(name=model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    best_metric = checkpoint.get('best_metric', 'N/A')
    return model.to(device).eval(), epoch, best_metric


def build_subsample(val_samples, patients_per_task, subsample_seed):
    """Patient-level subsample strictly from the held-out validation set."""
    by_task_patient = defaultdict(lambda: defaultdict(list))
    for s in val_samples:
        vol_key = get_sample_volume_key(s)
        by_task_patient[s['task']][vol_key].append(s)

    rng = random.Random(subsample_seed)
    selected_samples = []
    summary = []

    for task in DEFAULT_TASKS:
        patients = by_task_patient.get(task, {})
        patient_keys = sorted(patients.keys())
        rng.shuffle(patient_keys)
        n_avail = len(patient_keys)
        n_take = min(patients_per_task, n_avail) if patients_per_task > 0 else n_avail
        chosen = patient_keys[:n_take]

        n_slices = 0
        for pk in chosen:
            selected_samples.extend(patients[pk])
            n_slices += len(patients[pk])

        summary.append({
            'task': task,
            'n_patients_available_in_holdout': n_avail,
            'n_patients_selected': n_take,
            'n_slices_selected': n_slices,
        })

    return selected_samples, summary


def run_isolation_ablation(model, eval_samples, transform, image_size, device,
                           batch_size=16, base_tolerance_native_px=2.0):
    """
    Runs both conditions on each slice:
      1. Isolated: class-specific tight box prompt -> class GT
      2. No Isolation: merged foreground box prompt -> class GT
      3. Union Target: merged foreground box prompt -> union GT
    """
    # Results store:
    # results_isolated: task -> list of {'dice', 'iou', 'nsd', 'is_cooccurring'}
    # results_no_iso:   task -> list of {'dice', 'iou', 'nsd', 'is_cooccurring'}
    # results_union:    task -> list of {'dice', 'iou', 'nsd'}
    results_isolated = defaultdict(list)
    results_no_iso = defaultdict(list)
    results_union = defaultdict(list)

    n = len(eval_samples)

    for start in tqdm(range(0, n, batch_size), desc='Evaluating Isolation Ablation'):
        batch = eval_samples[start:start + batch_size]

        imgs_rgb_orig, labels_orig, tasks, orig_shapes, img_tensors = [], [], [], [], []
        for s in batch:
            cached = np.load(s['cached_path'], allow_pickle=False)
            img = cached['image'].astype(np.float32, copy=False)
            label = cached['label'].astype(np.float32, copy=False)
            img_rgb = np.stack([img] * 3, axis=-1)

            imgs_rgb_orig.append(img_rgb)
            labels_orig.append(label)
            tasks.append(s['task'])
            orig_shapes.append(img.shape)
            img_tensors.append(transform(image=img_rgb)['image'])

        img_batch = torch.stack(img_tensors).to(device)

        with torch.no_grad():
            image_embeddings_batch = model.image_encoder(img_batch)

        for i in range(len(batch)):
            task_name = tasks[i]
            label = labels_orig[i]
            img_rgb = imgs_rgb_orig[i]
            orig_h, orig_w = orig_shapes[i]
            resize_scale = image_size / max(orig_h, orig_w)
            effective_tolerance = base_tolerance_native_px * resize_scale
            img_embed = image_embeddings_batch[i:i + 1]

            foreground_labels = np.unique(label)
            foreground_labels = foreground_labels[foreground_labels > 0]
            if len(foreground_labels) == 0:
                continue

            is_cooccurring = (len(foreground_labels) > 1)

            # ------------------------------------------------------------------
            # Condition 2: No Isolation (Merged-Prompt)
            # ------------------------------------------------------------------
            merged_mask = (label > 0).astype(np.float32)
            transformed_merged = transform(image=img_rgb, mask=merged_mask)
            t_merged_mask = transformed_merged['mask']
            if t_merged_mask.ndim == 2:
                t_merged_mask = t_merged_mask.unsqueeze(0)
            merged_mask_np = t_merged_mask[0].numpy()
            target_merged = t_merged_mask.unsqueeze(0).float().to(device)

            merged_box = get_bbox_from_mask(merged_mask_np, margin=10)
            merged_box_tensor = torch.tensor([merged_box], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                sparse_m, dense_m = model.prompt_encoder(points=None, boxes=merged_bbox_tensor, masks=None)
                pred_mask_merged, _ = model.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_m,
                    dense_prompt_embeddings=dense_m,
                    multimask_output=False,
                )
                if pred_mask_merged.shape[-2:] != (image_size, image_size):
                    pred_mask_merged = F.interpolate(
                        pred_mask_merged.float(), size=(image_size, image_size),
                        mode='bilinear', align_corners=False,
                    )

            # Union evaluation
            dice_u = compute_dice_score(pred_mask_merged, target_merged)
            iou_u = compute_iou_score(pred_mask_merged, target_merged)
            nsd_u = compute_nsd_score(pred_mask_merged, target_merged, tolerance=effective_tolerance)
            results_union[task_name].append({'dice': dice_u, 'iou': iou_u, 'nsd': nsd_u})

            # ------------------------------------------------------------------
            # Condition 1: Proposed (Class-Specific Isolated Prompt) & Class GTs
            # ------------------------------------------------------------------
            for class_label in foreground_labels:
                class_label = int(class_label)
                isolated = (label == class_label).astype(np.float32)

                transformed_iso = transform(image=img_rgb, mask=isolated)
                t_mask = transformed_iso['mask']
                if t_mask.ndim == 2:
                    t_mask = t_mask.unsqueeze(0)
                mask_np = t_mask[0].numpy()
                target_class = t_mask.unsqueeze(0).float().to(device)

                clean_box = get_bbox_from_mask(mask_np, margin=10)
                bbox_tensor = torch.tensor([clean_box], dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    sparse_iso, dense_iso = model.prompt_encoder(points=None, boxes=bbox_tensor, masks=None)
                    pred_mask_iso, _ = model.mask_decoder(
                        image_embeddings=img_embed,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_iso,
                        dense_prompt_embeddings=dense_iso,
                        multimask_output=False,
                    )
                    if pred_mask_iso.shape[-2:] != (image_size, image_size):
                        pred_mask_iso = F.interpolate(
                            pred_mask_iso.float(), size=(image_size, image_size),
                            mode='bilinear', align_corners=False,
                        )

                # Proposed metrics
                dice_iso = compute_dice_score(pred_mask_iso, target_class)
                iou_iso = compute_iou_score(pred_mask_iso, target_class)
                nsd_iso = compute_nsd_score(pred_mask_iso, target_class, tolerance=effective_tolerance)

                # No-isolation metrics evaluated on the SAME target class
                dice_no_iso = compute_dice_score(pred_mask_merged, target_class)
                iou_no_iso = compute_iou_score(pred_mask_merged, target_class)
                nsd_no_iso = compute_nsd_score(pred_mask_merged, target_class, tolerance=effective_tolerance)

                item_iso = {'dice': dice_iso, 'iou': iou_iso, 'nsd': nsd_iso, 'is_cooccurring': is_cooccurring}
                item_no = {'dice': dice_no_iso, 'iou': iou_no_iso, 'nsd': nsd_no_iso, 'is_cooccurring': is_cooccurring}

                results_isolated[task_name].append(item_iso)
                results_no_iso[task_name].append(item_no)

    return results_isolated, results_no_iso, results_union


def aggregate_results(results_isolated, results_no_iso):
    task_summary = {}

    for task in DEFAULT_TASKS:
        iso_items = results_isolated.get(task, [])
        no_items = results_no_iso.get(task, [])
        if not iso_items:
            continue

        iso_dice = float(np.mean([x['dice'] for x in iso_items]))
        iso_iou = float(np.mean([x['iou'] for x in iso_items]))
        iso_nsd = float(np.mean([x['nsd'] for x in iso_items]))

        no_dice = float(np.mean([x['dice'] for x in no_items]))
        no_iou = float(np.mean([x['iou'] for x in no_items]))
        no_nsd = float(np.mean([x['nsd'] for x in no_items]))

        # Specifically co-occurring slices (where >= 2 foreground classes co-exist)
        co_iso_dice = float(np.mean([x['dice'] for x in iso_items if x['is_cooccurring']])) if any(x['is_cooccurring'] for x in iso_items) else iso_dice
        co_no_dice = float(np.mean([x['dice'] for x in no_items if x['is_cooccurring']])) if any(x['is_cooccurring'] for x in no_items) else no_dice

        task_summary[task] = {
            'iso_dice': iso_dice,
            'iso_iou': iso_iou,
            'iso_nsd': iso_nsd,
            'no_iso_dice': no_dice,
            'no_iso_iou': no_iou,
            'no_iso_nsd': no_nsd,
            'delta_dice': iso_dice - no_dice,
            'delta_iou': iso_iou - no_iou,
            'delta_nsd': iso_nsd - no_nsd,
            'cooccurring_iso_dice': co_iso_dice,
            'cooccurring_no_dice': co_no_dice,
            'cooccurring_delta_dice': co_iso_dice - co_no_dice,
            'n_instances': len(iso_items),
            'is_multi_class': task in MULTI_CLASS_TASKS,
        }

    # Macro averages
    macro_all = {
        'iso_dice': float(np.mean([v['iso_dice'] for v in task_summary.values()])),
        'no_iso_dice': float(np.mean([v['no_iso_dice'] for v in task_summary.values()])),
        'delta_dice': float(np.mean([v['delta_dice'] for v in task_summary.values()])),
        'iso_nsd': float(np.mean([v['iso_nsd'] for v in task_summary.values()])),
        'no_iso_nsd': float(np.mean([v['no_iso_nsd'] for v in task_summary.values()])),
        'delta_nsd': float(np.mean([v['delta_nsd'] for v in task_summary.values()])),
    }

    multi_tasks = [v for k, v in task_summary.items() if k in MULTI_CLASS_TASKS]
    macro_multi = {
        'iso_dice': float(np.mean([v['iso_dice'] for v in multi_tasks])),
        'no_iso_dice': float(np.mean([v['no_iso_dice'] for v in multi_tasks])),
        'delta_dice': float(np.mean([v['delta_dice'] for v in multi_tasks])),
        'iso_nsd': float(np.mean([v['iso_nsd'] for v in multi_tasks])),
        'no_iso_nsd': float(np.mean([v['no_iso_nsd'] for v in multi_tasks])),
        'delta_nsd': float(np.mean([v['delta_nsd'] for v in multi_tasks])),
    } if multi_tasks else {}

    single_tasks = [v for k, v in task_summary.items() if k in SINGLE_CLASS_TASKS]
    macro_single = {
        'iso_dice': float(np.mean([v['iso_dice'] for v in single_tasks])),
        'no_iso_dice': float(np.mean([v['no_iso_dice'] for v in single_tasks])),
        'delta_dice': float(np.mean([v['delta_dice'] for v in single_tasks])),
        'iso_nsd': float(np.mean([v['iso_nsd'] for v in single_tasks])),
        'no_iso_nsd': float(np.mean([v['no_iso_nsd'] for v in single_tasks])),
        'delta_nsd': float(np.mean([v['delta_nsd'] for v in single_tasks])),
    } if single_tasks else {}

    return task_summary, macro_all, macro_multi, macro_single


def print_tables(task_summary, macro_all, macro_multi, macro_single):
    print('\n' + '=' * 96)
    print('ABLATION STUDY: STOCHASTIC CLASS ISOLATION vs. UNISOLATED (MERGED) PROMPTING')
    print('-' * 96)
    print(f"{'Task':<22}{'Type':<12}{'Proposed Dice':>14}{'No-Iso Dice':>14}{'Delta Dice':>14}{'Delta NSD':>14}")
    print('-' * 96)
    for task in DEFAULT_TASKS:
        if task in task_summary:
            v = task_summary[task]
            ttype = 'Multi-Class' if v['is_multi_class'] else 'Single-Class'
            print(f"{task:<22}{ttype:<12}{v['iso_dice']*100:>13.2f}%{v['no_iso_dice']*100:>13.2f}%"
                  f"{v['delta_dice']*100:>+13.2f}%{v['delta_nsd']*100:>+13.2f}%")

    print('-' * 96)
    if macro_multi:
        print(f"{'Multi-Class Average':<22}{'6 Tasks':<12}{macro_multi['iso_dice']*100:>13.2f}%"
              f"{macro_multi['no_iso_dice']*100:>13.2f}%{macro_multi['delta_dice']*100:>+13.2f}%"
              f"{macro_multi['delta_nsd']*100:>+13.2f}%")
    if macro_single:
        print(f"{'Single-Class (Control)':<22}{'4 Tasks':<12}{macro_single['iso_dice']*100:>13.2f}%"
              f"{macro_single['no_iso_dice']*100:>13.2f}%{macro_single['delta_dice']*100:>+13.2f}%"
              f"{macro_single['delta_nsd']*100:>+13.2f}%")
    print(f"{'OVERALL MACRO':<22}{'10 Tasks':<12}{macro_all['iso_dice']*100:>13.2f}%"
          f"{macro_all['no_iso_dice']*100:>13.2f}%{macro_all['delta_dice']*100:>+13.2f}%"
          f"{macro_all['delta_nsd']*100:>+13.2f}%")
    print('=' * 96 + '\n')


def save_csv(task_summary, macro_all, macro_multi, macro_single, csv_path):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'task', 'task_type', 'proposed_isolated_dice', 'no_isolation_dice',
            'delta_dice', 'proposed_isolated_nsd', 'no_isolation_nsd', 'delta_nsd',
            'cooccurring_iso_dice', 'cooccurring_no_dice', 'cooccurring_delta_dice', 'n_instances'
        ])
        for task in DEFAULT_TASKS:
            if task in task_summary:
                v = task_summary[task]
                ttype = 'Multi-Class' if v['is_multi_class'] else 'Single-Class'
                writer.writerow([
                    task, ttype,
                    round(v['iso_dice'] * 100, 2), round(v['no_iso_dice'] * 100, 2),
                    round(v['delta_dice'] * 100, 2), round(v['iso_nsd'] * 100, 2),
                    round(v['no_iso_nsd'] * 100, 2), round(v['delta_nsd'] * 100, 2),
                    round(v['cooccurring_iso_dice'] * 100, 2), round(v['cooccurring_no_dice'] * 100, 2),
                    round(v['cooccurring_delta_dice'] * 100, 2), v['n_instances']
                ])
        if macro_multi:
            writer.writerow([
                'MULTI_CLASS_MACRO', 'Multi-Class',
                round(macro_multi['iso_dice'] * 100, 2), round(macro_multi['no_iso_dice'] * 100, 2),
                round(macro_multi['delta_dice'] * 100, 2), round(macro_multi['iso_nsd'] * 100, 2),
                round(macro_multi['no_iso_nsd'] * 100, 2), round(macro_multi['delta_nsd'] * 100, 2),
                '', '', '', ''
            ])
        if macro_single:
            writer.writerow([
                'SINGLE_CLASS_CONTROL', 'Single-Class',
                round(macro_single['iso_dice'] * 100, 2), round(macro_single['no_iso_dice'] * 100, 2),
                round(macro_single['delta_dice'] * 100, 2), round(macro_single['iso_nsd'] * 100, 2),
                round(macro_single['no_iso_nsd'] * 100, 2), round(macro_single['delta_nsd'] * 100, 2),
                '', '', '', ''
            ])
        writer.writerow([
            'OVERALL_MACRO', 'All-10',
            round(macro_all['iso_dice'] * 100, 2), round(macro_all['no_iso_dice'] * 100, 2),
            round(macro_all['delta_dice'] * 100, 2), round(macro_all['iso_nsd'] * 100, 2),
            round(macro_all['no_iso_nsd'] * 100, 2), round(macro_all['delta_nsd'] * 100, 2),
            '', '', '', ''
        ])
    print(f'Saved CSV: {csv_path}')


def plot_comparison_barplot(task_summary, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Group multi-class first, then single-class control
    ordered_tasks = [t for t in MULTI_CLASS_TASKS if t in task_summary] + \
                    [t for t in SINGLE_CLASS_TASKS if t in task_summary]

    short_names = [t.split('_', 1)[-1] for t in ordered_tasks]

    iso_dices = [task_summary[t]['iso_dice'] * 100 for t in ordered_tasks]
    no_dices = [task_summary[t]['no_iso_dice'] * 100 for t in ordered_tasks]

    x = np.arange(len(ordered_tasks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5))

    rects1 = ax.bar(x - width / 2, iso_dices, width, label='Proposed (Class-Isolated Prompt)', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x + width / 2, no_dices, width, label='Ablation (Merged / No Isolation)', color='#d62728', alpha=0.9)

    ax.set_ylabel('Dice Similarity Coefficient (DSC %)', fontsize=12)
    ax.set_title('Ablation Study: Effect of Stochastic Class Isolation on Anatomical Specificity', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Dividing line between multi-class and single-class control
    n_multi = len([t for t in MULTI_CLASS_TASKS if t in task_summary])
    if n_multi > 0 and n_multi < len(ordered_tasks):
        div_x = n_multi - 0.5
        ax.axvline(x=div_x, color='gray', linestyle=':', linewidth=1.5)
        ax.text(div_x / 2.0, 101, 'Multi-Class Tasks (Ablation Target)', ha='center', fontsize=10, fontweight='semibold', color='#333333')
        ax.text(div_x + (len(ordered_tasks) - div_x) / 2.0, 101, 'Single-Class Tasks (Control Group)', ha='center', fontsize=10, fontweight='semibold', color='#333333')

    ax.legend(loc='lower left', frameon=True, fontsize=10)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Figure: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Stochastic Class Isolation Ablation (no retraining)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model_dice.pth')
    parser.add_argument('--slice_cache_dir', type=str, required=True,
                        help='Path to precomputed MSD slice cache directory')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base dataset directory')
    parser.add_argument('--model', type=str, default='efficientvit-sam-l0')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='MUST match training holdout partition ratio')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='MUST match training holdout split seed')
    parser.add_argument('--patients_per_task', type=int, default=12,
                        help='Max patients sampled per task (0 means evaluate ALL holdout patients)')
    parser.add_argument('--subsample_seed', type=int, default=123,
                        help='Seed for patient subsampling (matches jitter ablation)')
    parser.add_argument('--base_tolerance_native_px', type=float, default=2.0,
                        help='Base NSD pixel tolerance before per-slice scaling')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./results/ablation_class_isolation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = MedicalDecathlonDataset(
        base_dir=args.base_dir, tasks=DEFAULT_TASKS, split='training',
        transform=None, balance_tasks=False, cached_slice_dir=args.slice_cache_dir,
    )
    _, val_samples, info = patient_level_holdout_split(
        dataset.samples, val_ratio=args.val_ratio, seed=args.split_seed
    )
    print(f"Full holdout: {info['n_val_slices']} slices from {info['n_val_volumes']} volumes")

    if args.patients_per_task > 0:
        eval_samples, subsample_summary = build_subsample(val_samples, args.patients_per_task, args.subsample_seed)
        total_p = sum(r['n_patients_selected'] for r in subsample_summary)
        total_s = sum(r['n_slices_selected'] for r in subsample_summary)
        print(f"Subsample: {total_p} patients, {total_s} slices selected.")
    else:
        eval_samples = val_samples
        subsample_summary = 'full_holdout'
        print(f"Using full holdout set ({len(eval_samples)} slices).")

    model, epoch, best_metric = load_finetuned_model(args.checkpoint, args.model, device)
    print(f'Loaded checkpoint from epoch: {epoch} (best metric: {best_metric})')

    transform = get_val_transforms(args.image_size)

    results_iso, results_no, results_u = run_isolation_ablation(
        model=model,
        eval_samples=eval_samples,
        transform=transform,
        image_size=args.image_size,
        device=device,
        batch_size=args.batch_size,
        base_tolerance_native_px=args.base_tolerance_native_px,
    )

    task_summary, macro_all, macro_multi, macro_single = aggregate_results(results_iso, results_no)
    print_tables(task_summary, macro_all, macro_multi, macro_single)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        'config': vars(args),
        'subsample_summary': subsample_summary,
        'task_summary': task_summary,
        'macro_all': macro_all,
        'macro_multi': macro_multi,
        'macro_single': macro_single,
    }
    with open(out_dir / 'class_isolation_ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved JSON: {out_dir / 'class_isolation_ablation_summary.json'}")

    save_csv(task_summary, macro_all, macro_multi, macro_single, out_dir / 'class_isolation_ablation_summary.csv')
    plot_comparison_barplot(task_summary, out_dir / 'class_isolation_ablation_barplot.png')

    print('\nAblation study completed successfully!')


if __name__ == '__main__':
    main()
