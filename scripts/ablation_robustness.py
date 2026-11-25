import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for p in [str(_repo_dir), str(_repo_dir / 'data'), str(_repo_dir / 'models'), str(_repo_dir / 'configs'), str(_repo_dir / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

"""
Ablation Study: Prompt-Box Robustness + NSD Tolerance Sensitivity
====================================================================
Runs on your ALREADY-TRAINED checkpoint. No retraining, no full-dataset
re-run. Produces TWO ablations from ONE pass over a deterministic subsample.

WHY THESE TWO
-------------
(A) Box-jitter prompt robustness
    Every number you've reported so far -- training and testing -- uses a
    box drawn exactly around the ground truth (+10px margin). That's a
    best-case "oracle" prompt. A real clinician's box/click will never be
    that precise. This ablation deliberately perturbs the box by several
    magnitudes and measures how much Dice/IoU degrade, giving you a
    defensible robustness curve instead of an untested assumption.

(B) NSD tolerance sensitivity sweep
    You already fixed NSD to scale its pixel tolerance by each slice's own
    resize factor. This sweep shows NSD across several tolerance choices
    (not just the one you settled on), so you have a transparent, principled
    answer if a reviewer asks "why this tolerance and not another?"

Ablation (B) is essentially FREE on top of (A): it reuses the prediction
already computed for the "0% jitter" (clean box) case in (A) and just
recomputes the NSD metric at different tolerances on that same prediction
-- no extra forward passes.

WHY A SUBSAMPLE, NOT THE FULL ~16,300-SLICE HOLDOUT
-----------------------------------------------------
Ablation (A) needs one decode pass PER JITTER LEVEL (default 4 levels), on
top of everything else. Running that on the full holdout would take
several hours for a diagnostic study that doesn't need that scale to be
trustworthy. Instead this script deterministically samples up to
`--patients_per_task` patients PER TASK from the SAME held-out patient
list used everywhere else in this project (same --val_ratio/--split_seed,
so it never touches a training patient). This keeps the patient-level
integrity of the whole project intact while bounding runtime to roughly
15-30 minutes depending on your GPU, instead of several hours.

State this explicitly in your methodology, e.g.:
    "Prompt-robustness and NSD-tolerance ablations were conducted on a
    deterministic, patient-level subsample of the held-out set (up to N
    patients per task, seed=<subsample_seed>) for computational
    efficiency; headline results in Table 1 use the full holdout set."
This is standard practice for ablation/robustness studies and does not
weaken the headline numbers, which are untouched.

The script also reports the subsample's own "clean box, standard
tolerance" Dice/IoU/NSD per task -- compare this against your full-dataset
holdout_evaluation_per_class.csv as a sanity check that the subsample is
representative before trusting the ablation curves.

USAGE
-----
python ablation_prompt_robustness_and_nsd_sweep.py \
    --checkpoint /path/to/best_model_dice.pth \
    --slice_cache_dir /path/to/msd_slice_cache_all10_new \
    --model efficientvit-sam-l0 \
    --image_size 512 \
    --val_ratio 0.2 \
    --split_seed 42 \
    --patients_per_task 12 \
    --batch_size 16 \
    --output_dir /path/to/results/ablation_prompt_nsd
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


# ---------------------------------------------------------------------------
# Small reused helpers (kept self-contained, consistent with the other scripts)
# ---------------------------------------------------------------------------

def get_bbox_from_mask(mask: np.ndarray, margin: int = 10) -> list:
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


def jitter_box(box, frac, rng, img_w, img_h):
    """frac=0.0 returns the box unchanged (the 'clean' baseline). frac=0.10
    means each edge can move by up to +/-10% of that box's own width/height."""
    if frac <= 0:
        return list(box)
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    nx1 = x1 + rng.uniform(-frac, frac) * w
    ny1 = y1 + rng.uniform(-frac, frac) * h
    nx2 = x2 + rng.uniform(-frac, frac) * w
    ny2 = y2 + rng.uniform(-frac, frac) * h
    nx1, nx2 = sorted([nx1, nx2])
    ny1, ny2 = sorted([ny1, ny2])
    nx1 = max(0.0, min(nx1, img_w - 2))
    ny1 = max(0.0, min(ny1, img_h - 2))
    nx2 = max(nx1 + 1.0, min(nx2, img_w))
    ny2 = max(ny1 + 1.0, min(ny2, img_h))
    return [nx1, ny1, nx2, ny2]


# ---------------------------------------------------------------------------
# Deterministic, patient-level subsample
# ---------------------------------------------------------------------------

def build_subsample(val_samples, patients_per_task, subsample_seed):
    by_task_patient = defaultdict(lambda: defaultdict(list))
    for s in val_samples:
        vol_key = get_sample_volume_key(s)
        by_task_patient[s['task']][vol_key].append(s)

    rng = random.Random(subsample_seed)
    selected_samples = []
    summary = []

    for task in DEFAULT_TASKS:
        patients = by_task_patient.get(task, {})
        patient_keys = sorted(patients.keys())  # deterministic base order before shuffling
        rng.shuffle(patient_keys)
        n_avail = len(patient_keys)
        n_take = min(patients_per_task, n_avail)
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


def print_subsample_summary(summary):
    print('\n' + '=' * 80)
    print('DETERMINISTIC SUBSAMPLE (patient-level, drawn from the SAME held-out set)')
    print('-' * 80)
    print(f"{'Task':<24}{'Patients available':>20}{'Patients used':>16}{'Slices used':>16}")
    for row in summary:
        print(f"{row['task']:<24}{row['n_patients_available_in_holdout']:>20}"
              f"{row['n_patients_selected']:>16}{row['n_slices_selected']:>16}")
    total_patients = sum(r['n_patients_selected'] for r in summary)
    total_slices = sum(r['n_slices_selected'] for r in summary)
    print('-' * 80)
    print(f"TOTAL: {total_patients} patients, {total_slices} slices selected for this ablation")
    print('=' * 80 + '\n')


# ---------------------------------------------------------------------------
# Core ablation loop -- ONE encoder pass per slice, reused across every
# class and every jitter level for that slice.
# ---------------------------------------------------------------------------

def run_ablation(model, subsample, transform, image_size, device, batch_size,
                  jitter_levels, nsd_tolerances_native_px, base_tolerance_native_px,
                  jitter_seed):
    jitter_results = defaultdict(lambda: defaultdict(list))     # level -> task -> [{'dice','iou'}]
    nsd_sweep_results = defaultdict(lambda: defaultdict(list))  # tol_native -> task -> [nsd]
    standard_results = defaultdict(list)                        # task -> [{'dice','iou','nsd'}]  (clean box, standard tolerance -- for cross-checking vs. the full-dataset run)

    jitter_rng = random.Random(jitter_seed)
    n = len(subsample)

    for start in tqdm(range(0, n, batch_size), desc='Running ablation (jitter + NSD sweep)'):
        batch = subsample[start:start + batch_size]

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

            for class_label in foreground_labels:
                class_label = int(class_label)
                isolated = (label == class_label).astype(np.float32)

                transformed = transform(image=img_rgb, mask=isolated)
                t_mask = transformed['mask']
                if t_mask.ndim == 2:
                    t_mask = t_mask.unsqueeze(0)
                mask_np = t_mask[0].numpy()
                target = t_mask.unsqueeze(0).float().to(device)

                clean_box = get_bbox_from_mask(mask_np)

                for level in jitter_levels:
                    jittered_box = jitter_box(clean_box, level, jitter_rng, image_size, image_size)
                    bbox_tensor = torch.tensor([jittered_box], dtype=torch.float32).unsqueeze(0).to(device)

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

                    dice = compute_dice_score(pred_mask, target)
                    iou = compute_iou_score(pred_mask, target)
                    jitter_results[level][task_name].append({'dice': dice, 'iou': iou})

                    # Piggy-back the NSD sweep on the CLEAN box pass only -- zero extra forward passes.
                    if level == 0.0:
                        nsd_standard = compute_nsd_score(pred_mask, target, tolerance=effective_tolerance)
                        standard_results[task_name].append({'dice': dice, 'iou': iou, 'nsd': nsd_standard})

                        for tol_native in nsd_tolerances_native_px:
                            tol_scaled = tol_native * resize_scale
                            nsd_val = compute_nsd_score(pred_mask, target, tolerance=tol_scaled)
                            nsd_sweep_results[tol_native][task_name].append(nsd_val)

    return jitter_results, nsd_sweep_results, standard_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_jitter(jitter_results):
    summary = {}
    for level, per_task in jitter_results.items():
        task_means = {}
        for task, items in per_task.items():
            task_means[task] = {
                'dice': float(np.mean([x['dice'] for x in items])),
                'iou': float(np.mean([x['iou'] for x in items])),
                'n': len(items),
            }
        macro_dice = float(np.mean([v['dice'] for v in task_means.values()])) if task_means else float('nan')
        macro_iou = float(np.mean([v['iou'] for v in task_means.values()])) if task_means else float('nan')
        summary[level] = {'per_task': task_means, 'macro_dice': macro_dice, 'macro_iou': macro_iou}
    return summary


def aggregate_nsd_sweep(nsd_sweep_results):
    summary = {}
    for tol, per_task in nsd_sweep_results.items():
        task_means = {task: float(np.mean(vals)) for task, vals in per_task.items()}
        macro_nsd = float(np.mean(list(task_means.values()))) if task_means else float('nan')
        summary[tol] = {'per_task': task_means, 'macro_nsd': macro_nsd}
    return summary


def aggregate_standard(standard_results):
    out = {}
    for task, items in standard_results.items():
        out[task] = {
            'dice': float(np.mean([x['dice'] for x in items])),
            'iou': float(np.mean([x['iou'] for x in items])),
            'nsd': float(np.mean([x['nsd'] for x in items])),
            'n': len(items),
        }
    return out


# ---------------------------------------------------------------------------
# Printing / saving
# ---------------------------------------------------------------------------

def print_jitter_table(jitter_summary):
    print('\n' + '=' * 80)
    print('(A) BOX-JITTER PROMPT ROBUSTNESS  (macro-average across 10 tasks)')
    print('-' * 80)
    print(f"{'Jitter level':<20}{'Macro Dice':>15}{'Macro IoU':>15}")
    for level in sorted(jitter_summary.keys()):
        s = jitter_summary[level]
        print(f"{f'{int(level*100)}%':<20}{s['macro_dice']*100:>14.2f}%{s['macro_iou']*100:>14.2f}%")
    print('=' * 80 + '\n')


def print_nsd_sweep_table(nsd_summary):
    print('\n' + '=' * 80)
    print('(B) NSD TOLERANCE SENSITIVITY SWEEP  (macro-average across 10 tasks)')
    print('-' * 80)
    print(f"{'Tolerance (native px)':<24}{'Macro NSD':>15}")
    for tol in sorted(nsd_summary.keys()):
        print(f"{tol:<24}{nsd_summary[tol]['macro_nsd']*100:>14.2f}%")
    print('=' * 80 + '\n')


def print_standard_crosscheck(standard_agg):
    print('\n' + '=' * 80)
    print('SUBSAMPLE CROSS-CHECK  (clean box, standard tolerance -- compare against')
    print('your FULL-dataset holdout_evaluation_per_class.csv to sanity-check that')
    print('this subsample is representative)')
    print('-' * 80)
    print(f"{'Task':<24}{'Dice':>10}{'IoU':>10}{'NSD':>10}{'N':>8}")
    for task in DEFAULT_TASKS:
        if task in standard_agg:
            s = standard_agg[task]
            print(f"{task:<24}{s['dice']*100:>9.2f}%{s['iou']*100:>9.2f}%{s['nsd']*100:>9.2f}%{s['n']:>8}")
    print('=' * 80 + '\n')


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {path}')


def save_jitter_csv(jitter_summary, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['jitter_level_pct', 'task', 'dice', 'iou', 'n'])
        for level in sorted(jitter_summary.keys()):
            for task, v in jitter_summary[level]['per_task'].items():
                writer.writerow([int(level * 100), task, v['dice'], v['iou'], v['n']])
            writer.writerow([int(level * 100), 'OVERALL_MACRO', jitter_summary[level]['macro_dice'], jitter_summary[level]['macro_iou'], ''])
    print(f'Saved: {path}')


def save_nsd_sweep_csv(nsd_summary, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tolerance_native_px', 'task', 'nsd'])
        for tol in sorted(nsd_summary.keys()):
            for task, v in nsd_summary[tol]['per_task'].items():
                writer.writerow([tol, task, v])
            writer.writerow([tol, 'OVERALL_MACRO', nsd_summary[tol]['macro_nsd']])
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_jitter_curve(jitter_summary, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    levels = sorted(jitter_summary.keys())
    macro_dice = [jitter_summary[l]['macro_dice'] for l in levels]
    macro_iou = [jitter_summary[l]['macro_iou'] for l in levels]
    x_labels = [f'{int(l * 100)}%' for l in levels]

    plt.figure(figsize=(7, 5))
    plt.plot(x_labels, macro_dice, marker='o', linewidth=2, label='Dice')
    plt.plot(x_labels, macro_iou, marker='s', linewidth=2, label='IoU')
    plt.xlabel('Box prompt jitter magnitude (% of box size)')
    plt.ylabel('Score (macro-average across 10 tasks)')
    plt.title('Prompt-Box Robustness: Score vs. Prompt Imprecision')
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_nsd_sweep(nsd_summary, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tolerances = sorted(nsd_summary.keys())
    plt.figure(figsize=(9, 6))
    for task in DEFAULT_TASKS:
        if all(task in nsd_summary[t]['per_task'] for t in tolerances):
            vals = [nsd_summary[t]['per_task'][task] for t in tolerances]
            plt.plot(tolerances, vals, marker='o', linewidth=1.2, alpha=0.8, label=task.split('_', 1)[-1])
    macro_vals = [nsd_summary[t]['macro_nsd'] for t in tolerances]
    plt.plot(tolerances, macro_vals, marker='D', linewidth=3, color='black', label='Overall (macro)')
    plt.xlabel('NSD tolerance (native pixels, scaled per slice)')
    plt.ylabel('NSD')
    plt.title('NSD Sensitivity to Tolerance Choice')
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Prompt-robustness + NSD-tolerance ablation (no retraining)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--slice_cache_dir', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--model', type=str, default='efficientvit-sam-l0')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='MUST match training')
    parser.add_argument('--split_seed', type=int, default=42, help='MUST match training')
    parser.add_argument('--patients_per_task', type=int, default=12,
                         help='Max patients sampled per task for this ablation (uses all if a task has fewer)')
    parser.add_argument('--subsample_seed', type=int, default=123)
    parser.add_argument('--jitter_seed', type=int, default=777)
    parser.add_argument('--jitter_levels_pct', type=str, default='0,10,20,30',
                         help='Comma-separated jitter magnitudes as percent of box size')
    parser.add_argument('--nsd_tolerances_native_px', type=str, default='1,2,3,5,10',
                         help='Comma-separated NSD tolerance values to sweep, in native (pre-resize) pixels')
    parser.add_argument('--nsd_base_tolerance_native_px', type=float, default=2.0,
                         help='The standard tolerance used elsewhere in the project, for the cross-check row')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    jitter_levels = sorted(set([float(x) / 100.0 for x in args.jitter_levels_pct.split(',')]))
    nsd_tolerances = sorted(set([float(x) for x in args.nsd_tolerances_native_px.split(',')]))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Jitter levels: {[f"{int(l*100)}%" for l in jitter_levels]}')
    print(f'NSD tolerance sweep (native px): {nsd_tolerances}')

    dataset = MedicalDecathlonDataset(
        base_dir=args.base_dir, tasks=DEFAULT_TASKS, split='training',
        transform=None, balance_tasks=False, cached_slice_dir=args.slice_cache_dir,
    )
    _, val_samples, info = patient_level_holdout_split(
        dataset.samples, val_ratio=args.val_ratio, seed=args.split_seed
    )
    print(f"Full holdout: {info['n_val_slices']} slices from {info['n_val_volumes']} volumes")

    subsample, subsample_summary = build_subsample(val_samples, args.patients_per_task, args.subsample_seed)
    print_subsample_summary(subsample_summary)

    model, epoch, best_metric = load_finetuned_model(args.checkpoint, args.model, device)
    print(f'Model epoch: {epoch} | training best metric: {best_metric}')

    transform = get_val_transforms(args.image_size)

    jitter_results, nsd_sweep_results, standard_results = run_ablation(
        model, subsample, transform, args.image_size, device, args.batch_size,
        jitter_levels, nsd_tolerances, args.nsd_base_tolerance_native_px, args.jitter_seed,
    )

    jitter_summary = aggregate_jitter(jitter_results)
    nsd_summary = aggregate_nsd_sweep(nsd_sweep_results)
    standard_agg = aggregate_standard(standard_results)

    print_jitter_table(jitter_summary)
    print_nsd_sweep_table(nsd_summary)
    print_standard_crosscheck(standard_agg)

    out_dir = Path(args.output_dir)

    save_json({
        'config': vars(args),
        'subsample_summary': subsample_summary,
        'jitter_summary': jitter_summary,
    }, out_dir / 'jitter_robustness_results.json')
    save_jitter_csv(jitter_summary, out_dir / 'jitter_robustness_results.csv')

    save_json({
        'config': vars(args),
        'nsd_summary': nsd_summary,
    }, out_dir / 'nsd_tolerance_sweep_results.json')
    save_nsd_sweep_csv(nsd_summary, out_dir / 'nsd_tolerance_sweep_results.csv')

    save_json({'standard_crosscheck': standard_agg}, out_dir / 'subsample_standard_crosscheck.json')

    plot_jitter_curve(jitter_summary, out_dir / 'jitter_robustness_curve.png')
    plot_nsd_sweep(nsd_summary, out_dir / 'nsd_tolerance_sweep.png')

    print('\nAll done. Compare subsample_standard_crosscheck.json against your full-dataset '
          'holdout_evaluation_per_class.csv -- they should be reasonably close per task, '
          'confirming the subsample used for this ablation is representative.')


if __name__ == '__main__':
    main()
