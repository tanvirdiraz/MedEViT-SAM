"""
Medical Decathlon Dataset for Multi-Task Fine-Tuning

Loads and prepares data from all 5 Medical Decathlon tasks
with data augmentation and balanced sampling.

Includes volume-level caching to avoid reloading the same 3D NIfTI
files multiple times during data loading.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from collections import OrderedDict, defaultdict
import psutil
import os
import json


class VolumeCache:
    """LRU cache for 3D NIfTI volumes to avoid repeated disk I/O
    
    Caches loaded volumes in memory with automatic eviction when
    memory usage exceeds a threshold.
    """
    
    def __init__(self, max_memory_gb: float = 8.0):
        """
        Args:
            max_memory_gb: Maximum memory to use for caching (in GB)
        """
        self.cache = OrderedDict()
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, filepath: str) -> np.ndarray:
        """Get volume from cache or load from disk
        
        Args:
            filepath: Path to NIfTI file
            
        Returns:
            Loaded volume as numpy array
        """
        filepath = str(filepath)
        
        if filepath in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(filepath)
            self.hits += 1
            return self.cache[filepath]
        
        # Not in cache, load from disk
        self.misses += 1
        nii = nib.load(filepath)
        data = nii.get_fdata(dtype=np.float32)
        
        # Add to cache
        data_size = data.nbytes
        self.cache[filepath] = data
        self.current_size_bytes += data_size
        
        # Evict LRU items if over memory limit
        while self.current_size_bytes > self.max_memory_bytes and len(self.cache) > 1:
            evicted_path, evicted_data = self.cache.popitem(last=False)
            self.current_size_bytes -= evicted_data.nbytes
        
        return data
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_accesses = self.hits + self.misses
        hit_rate = (self.hits / total_accesses * 100) if total_accesses > 0 else 0
        size_gb = self.current_size_bytes / (1024 ** 3)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cached_volumes': len(self.cache),
            'current_size_gb': size_gb,
            'max_size_gb': self.max_memory_bytes / (1024 ** 3)
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.current_size_bytes = 0

    def set_max_memory_gb(self, max_memory_gb: float):
        """Update cache memory limit and evict if needed."""
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        while self.current_size_bytes > self.max_memory_bytes and len(self.cache) > 1:
            _, evicted_data = self.cache.popitem(last=False)
            self.current_size_bytes -= evicted_data.nbytes


# Global volume cache instance (shared across all dataset instances)
_VOLUME_CACHE = VolumeCache(max_memory_gb=8.0)


def _strip_nii_gz(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    return Path(filename).stem


def get_sample_volume_key(sample: Dict) -> str:
    """Return a stable volume identifier for patient-level splitting."""
    task_name = sample.get('task', 'unknown_task')

    if sample.get('volume_id'):
        return f"{task_name}::{sample['volume_id']}"

    if 'img_path' in sample:
        volume_id = _strip_nii_gz(Path(sample['img_path']).name)
        return f"{task_name}::{volume_id}"

    cached_path = sample.get('cached_path')
    if cached_path:
        name = Path(cached_path).name
        if name.endswith('.npz'):
            name = name[:-4]
        if '__z' in name:
            name = name.rsplit('__z', 1)[0]
        parts = name.split('__')
        if len(parts) >= 2:
            task_name = parts[0]
            volume_id = parts[1]
            return f"{task_name}::{volume_id}"
        return f"{task_name}::{name}"

    return f"{task_name}::slice_{sample.get('slice_idx', 0)}"


def balance_samples_by_task(samples: List[Dict], tasks: List[str]) -> List[Dict]:
    """Oversample smaller tasks to match the largest task in the training split."""
    if not samples:
        return samples

    task_to_samples = {task: [] for task in tasks}
    for sample in samples:
        task_to_samples.setdefault(sample['task'], []).append(sample)

    available_counts = [len(task_samples) for task_samples in task_to_samples.values() if task_samples]
    if not available_counts:
        return samples

    max_count = max(available_counts)
    balanced_samples = []

    print("\nTraining task distribution before balancing:")
    for task in tasks:
        count = len(task_to_samples.get(task, []))
        print(f"  {task}: {count} slices")

    for task in tasks:
        task_samples = list(task_to_samples.get(task, []))
        if not task_samples:
            continue

        while len(task_samples) < max_count:
            task_samples.append(random.choice(task_samples))

        balanced_samples.extend(task_samples[:max_count])

    random.shuffle(balanced_samples)

    print("\nTraining task distribution after balancing:")
    balanced_counts = {}
    for sample in balanced_samples:
        task = sample['task']
        balanced_counts[task] = balanced_counts.get(task, 0) + 1
    for task, count in balanced_counts.items():
        print(f"  {task}: {count} slices")

    return balanced_samples


def patient_level_holdout_split(samples: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    """Split samples by volume so slices from one scan never cross train/val."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    volume_to_samples = defaultdict(list)
    for sample in samples:
        volume_to_samples[get_sample_volume_key(sample)].append(sample)

    volume_keys = list(volume_to_samples.keys())
    if len(volume_keys) < 2:
        raise ValueError("Need at least 2 volumes for patient-level holdout")

    rng = random.Random(seed)
    rng.shuffle(volume_keys)

    n_val_volumes = max(1, int(round(len(volume_keys) * val_ratio)))
    n_val_volumes = min(n_val_volumes, len(volume_keys) - 1)
    val_keys = set(volume_keys[:n_val_volumes])

    train_samples = []
    val_samples = []
    for key, grouped_samples in volume_to_samples.items():
        if key in val_keys:
            val_samples.extend(grouped_samples)
        else:
            train_samples.extend(grouped_samples)

    info = {
        'split_mode': 'patient',
        'seed': seed,
        'val_ratio': val_ratio,
        'num_folds': 1,
        'fold_index': 0,
        'n_total_volumes': len(volume_keys),
        'n_train_volumes': len(volume_keys) - len(val_keys),
        'n_val_volumes': len(val_keys),
        'n_train_slices': len(train_samples),
        'n_val_slices': len(val_samples),
    }
    return train_samples, val_samples, info


def patient_level_kfold_split(samples: List[Dict], num_folds: int, fold_index: int, seed: int = 42):
    """Create one patient-level fold for cross-validation."""
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for k-fold split")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("fold_index must be in [0, num_folds)")

    volume_to_samples = defaultdict(list)
    for sample in samples:
        volume_to_samples[get_sample_volume_key(sample)].append(sample)

    volume_keys = list(volume_to_samples.keys())
    if len(volume_keys) < num_folds:
        raise ValueError(f"Need at least {num_folds} volumes for {num_folds}-fold CV")

    rng = random.Random(seed)
    rng.shuffle(volume_keys)

    folds = [[] for _ in range(num_folds)]
    for idx, key in enumerate(volume_keys):
        folds[idx % num_folds].append(key)

    val_keys = set(folds[fold_index])
    train_samples = []
    val_samples = []
    for key, grouped_samples in volume_to_samples.items():
        if key in val_keys:
            val_samples.extend(grouped_samples)
        else:
            train_samples.extend(grouped_samples)

    info = {
        'split_mode': 'patient_kfold',
        'seed': seed,
        'val_ratio': None,
        'num_folds': num_folds,
        'fold_index': fold_index,
        'n_total_volumes': len(volume_keys),
        'n_train_volumes': len(volume_keys) - len(val_keys),
        'n_val_volumes': len(val_keys),
        'n_train_slices': len(train_samples),
        'n_val_slices': len(val_samples),
    }
    return train_samples, val_samples, info


class MedicalDecathlonSubset(Dataset):
    """Subset wrapper with independent transform and optional train-only balancing."""

    def __init__(self, base_dataset, samples: List[Dict], transform=None, balance_tasks: bool = False):
        self.base_dataset = base_dataset
        self.tasks = list(base_dataset.tasks)
        self.transform = transform
        self.samples = [dict(sample) for sample in samples]

        if balance_tasks:
            self.samples = balance_samples_by_task(self.samples, self.tasks)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.base_dataset._build_item_from_sample(self.samples[idx], transform=self.transform)


class MedicalDecathlonDataset(Dataset):
    """Multi-task dataset for all Medical Decathlon tasks"""
    
    def __init__(self, base_dir: str, tasks: List[str], split: str = 'training',
                 transform=None, max_slices_per_volume: int = None,
                 balance_tasks: bool = True,
                 cached_slice_dir: Optional[str] = None):
        """
        Args:
            base_dir: Base directory containing all tasks
            tasks: List of task names (e.g., ['Task02_Heart', 'Task04_Hippocampus'])
            split: 'training' or 'test'
            transform: Augmentation transforms
            max_slices_per_volume: Limit slices per volume (for memory)
            balance_tasks: Balance samples across tasks
        """
        self.base_dir = Path(base_dir)
        self.tasks = tasks
        self.split = split
        self.transform = transform
        self.max_slices_per_volume = max_slices_per_volume
        self._cache_file = self.base_dir / ".slice_index_cache.json"
        self.cached_slice_dir = Path(cached_slice_dir) if cached_slice_dir else None
        
        # Load all slices from all tasks
        self.samples = []
        self._load_all_samples()
        
        # Balance tasks if requested
        if balance_tasks:
            self._balance_tasks()
        
        print(f"Loaded {len(self.samples)} slices from {len(tasks)} tasks")
    
    def _load_all_samples(self):
        """Load all valid slices from all tasks"""
        if self.cached_slice_dir and self._try_load_precomputed_slice_index():
            return

        if self._try_load_index_cache():
            return
        
        for task_name in self.tasks:
            task_dir = self.base_dir / task_name
            
            if not task_dir.exists():
                print(f"Warning: Task directory not found: {task_dir}")
                continue
            
            # Get file list
            images_dir = task_dir / 'imagesTr'
            labels_dir = task_dir / 'labelsTr'
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: Missing data directories for {task_name}")
                continue
            
            # Get all image files (skip macOS metadata files)
            image_files = sorted([f for f in images_dir.glob('*.nii.gz') 
                                 if not f.name.startswith('._')])
            
            for img_path in image_files:
                # Corresponding label file
                label_path = labels_dir / img_path.name
                
                if not label_path.exists():
                    continue
                
                # Load volume to extract slices with annotations
                try:
                    label_nii = nib.load(str(label_path))
                    label_data = label_nii.get_fdata(dtype=np.float32)
                    slices_added_for_volume = 0

                    # Find slices with annotations
                    for z in range(label_data.shape[2]):
                        if label_data[:, :, z].sum() > 0:  # Has annotation
                            self.samples.append({
                                'task': task_name,
                                'img_path': str(img_path),
                                'label_path': str(label_path),
                                'slice_idx': z,
                                'task_id': self.tasks.index(task_name)
                            })
                            slices_added_for_volume += 1
                            
                            # Limit slices per volume
                            if (self.max_slices_per_volume and 
                                slices_added_for_volume >= self.max_slices_per_volume):
                                break
                
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

        self._save_index_cache()

    def _cache_signature(self):
        """Build a lightweight signature to validate cache compatibility."""
        return {
            'base_dir': str(self.base_dir),
            'tasks': list(self.tasks),
            'max_slices_per_volume': self.max_slices_per_volume,
        }

    def _try_load_precomputed_slice_index(self):
        """Load precomputed 2D slice index from local cache directory."""
        index_path = self.cached_slice_dir / "slice_index.json"
        if not index_path.exists():
            print(f"Precomputed slice index not found: {index_path}")
            return False

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            raw_samples = payload.get('samples', [])
            if not raw_samples:
                return False

            enabled_tasks = set(self.tasks)
            filtered = []
            for sample in raw_samples:
                task_name = sample.get('task')
                cached_path = sample.get('cached_path')
                if task_name not in enabled_tasks or not cached_path:
                    continue

                abs_cached_path = Path(cached_path)
                if not abs_cached_path.is_absolute():
                    abs_cached_path = (self.cached_slice_dir / cached_path).resolve()

                if not abs_cached_path.exists():
                    continue

                filtered.append({
                    'task': task_name,
                    'cached_path': str(abs_cached_path),
                    'slice_idx': int(sample.get('slice_idx', 0)),
                    'task_id': self.tasks.index(task_name)
                })

            if not filtered:
                return False

            self.samples = filtered
            print(f"Loaded {len(self.samples)} precomputed slices from cache: {index_path}")
            return True
        except Exception as e:
            print(f"Warning: failed to load precomputed slice cache ({e}).")
            return False

    def _try_load_index_cache(self):
        """Load precomputed slice index if available and compatible."""
        if not self._cache_file.exists():
            return False

        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            if payload.get('signature') != self._cache_signature():
                return False

            cached_samples = payload.get('samples', [])
            if not cached_samples:
                return False

            self.samples = cached_samples
            print(f"Loaded {len(self.samples)} indexed slices from cache: {self._cache_file}")
            return True
        except Exception as e:
            print(f"Warning: failed to load slice index cache ({e}). Rebuilding index...")
            return False

    def _save_index_cache(self):
        """Persist slice index for faster subsequent startup."""
        try:
            payload = {
                'signature': self._cache_signature(),
                'samples': self.samples,
            }
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"Warning: failed to save slice index cache ({e}).")
    
    def _balance_tasks(self):
        """Balance samples across tasks by oversampling smaller tasks"""
        self.samples = balance_samples_by_task(self.samples, self.tasks)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single slice with augmentation"""
        return self._build_item_from_sample(self.samples[idx], transform=self.transform)

    def _build_item_from_sample(self, sample, transform=None):
        """Build model-ready item from one indexed slice sample."""

        if 'cached_path' in sample:
            cached = np.load(sample['cached_path'], allow_pickle=False)
            img_slice = cached['image'].astype(np.float32, copy=False)
            label_slice = cached['label'].astype(np.float32, copy=False)
        else:
            # Load slice from volume using cache (avoids repeated disk I/O)
            img_data = _VOLUME_CACHE.get(sample['img_path'])
            label_data = _VOLUME_CACHE.get(sample['label_path'])

            if img_data.ndim == 4:
                img_data = img_data[..., 0]

            # Extract slice
            img_slice = img_data[:, :, sample['slice_idx']]
            label_slice = label_data[:, :, sample['slice_idx']]

            # Normalize image
            img_slice = self._normalize_image(img_slice)
        
        # Convert to 3-channel for SAM
        img_slice_rgb = np.stack([img_slice] * 3, axis=-1)
        
        # --- GENERALIZED MULTI-CLASS ISOLATION ---
        # 1. Find all unique structures that actually exist in this specific slice
        labels_in_slice = np.unique(label_slice)

        # 2. Filter out the background (Label 0)
        foreground_labels = labels_in_slice[labels_in_slice > 0]

        if len(foreground_labels) > 0:
            # 3. Randomly select ONE specific class to teach SAM for this step
            # (e.g., it picks 1 for Edema, or 3 for Enhancing Core)
            target_label = np.random.choice(foreground_labels)

            # 4. Isolate ONLY that selected class. Everything else becomes 0.
            # We overwrite the 'label_slice' variable so the rest of this method
            # (bounding box generation, etc.) uses this isolated shape automatically.
            label_slice = (label_slice == target_label).astype(np.float32)
        else:
            # This is an empty background slice
            label_slice = np.zeros_like(label_slice, dtype=np.float32)
        # -----------------------------------------
        
        # Apply transforms
        active_transform = transform if transform is not None else self.transform
        if active_transform:
            transformed = active_transform(image=img_slice_rgb, mask=label_slice)
            img_slice_rgb = transformed['image']
            label_slice = transformed['mask']
            
            # Ensure mask has channel dimension after transforms
            if label_slice.ndim == 2:
                label_slice = label_slice.unsqueeze(0)
            
            # Generate prompt (bounding box from ground truth)
            bbox = self._get_bbox_from_mask(label_slice[0].numpy())
        else:
            # Default: convert to tensor
            img_slice_rgb = torch.from_numpy(img_slice_rgb).permute(2, 0, 1).float()
            label_slice_tensor = torch.from_numpy(label_slice).unsqueeze(0).float()
            
            # Generate prompt (bounding box from ground truth)
            bbox = self._get_bbox_from_mask(label_slice)
            label_slice = label_slice_tensor
        
        return {
            'image': img_slice_rgb,
            'mask': label_slice,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'task_id': sample['task_id'],
            'task_name': sample['task']
        }
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity"""
        # Clip outliers
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image.astype(np.float32)
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """Extract bounding box from mask"""
        # Ensure mask is 2D
        if mask.ndim == 1:
            h = w = int(np.sqrt(len(mask)))
            mask = mask.reshape(h, w)
        
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0:
            # Empty mask - return small bbox at center
            h, w = mask.shape
            return [w//2 - 10, h//2 - 10, w//2 + 10, h//2 + 10]
        
        # Add margin
        margin = 10
        x_min = max(0, xs.min() - margin)
        y_min = max(0, ys.min() - margin)
        x_max = min(mask.shape[1], xs.max() + margin)
        y_max = min(mask.shape[0], ys.max() + margin)
        
        return [x_min, y_min, x_max, y_max]


def get_train_transforms(image_size=1024, fast_mode: bool = False):
    """Get training augmentation pipeline.

    Uses longest-edge resize + symmetric zero-pad (never squash) so the model
    always receives a true image_size x image_size tensor with correct aspect ratio.
    This is required for the static ONNX decoder which expects fixed-size embeddings.
    """

    if fast_mode:
        return A.Compose([
            # Scale longest edge to image_size (linear interp), preserve aspect ratio
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            # Pad shorter edge symmetrically with constant black — Albumentations v2 API
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0, fill_mask=0,
                position='center'
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return A.Compose([
        # Scale longest edge to image_size, preserve aspect ratio (no squash)
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        # Pad shorter edge symmetrically with constant black — Albumentations v2 API
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0, fill_mask=0,
            position='center'
        ),

        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Affine replaces deprecated ShiftScaleRotate in Albumentations v2
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.8, 1.2),
            rotate=(-30, 30),
            mode=cv2.BORDER_CONSTANT,
            cval=0, cval_mask=0,
            p=0.5
        ),

        # Elastic deformation (common in medical imaging)
        # alpha_affine removed in Albumentations v2; fill/fill_mask replace value/mask_value
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0, fill_mask=0,
            p=0.3
        ),

        # Intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # var_limit renamed to std_range in Albumentations v2 (values are std, not variance)
        A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # Standardize with ImageNet stats (after [0,1] normalization by A.Normalize)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size=1024):
    """Get validation transforms (no augmentation).

    Mirrors the same resize+pad strategy as training so val images have
    identical spatial layout to what the ONNX decoder expects.
    """
    return A.Compose([
        # Scale longest edge to image_size, preserve aspect ratio
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        # Pad shorter edge symmetrically with constant black — Albumentations v2 API
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0, fill_mask=0,
            position='center'
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_volume_cache_stats() -> Dict:
    """Get cache statistics for diagnostics
    
    Returns:
        Dictionary with cache hit rate, size, etc.
    """
    return _VOLUME_CACHE.get_stats()


def clear_volume_cache():
    """Clear the volume cache (useful between epochs or runs)"""
    _VOLUME_CACHE.clear()


def configure_volume_cache(max_memory_gb: float):
    """Configure max RAM used by the volume cache."""
    if max_memory_gb <= 0:
        raise ValueError("max_memory_gb must be > 0")
    _VOLUME_CACHE.set_max_memory_gb(max_memory_gb)
    print(f"Volume cache limit set to {max_memory_gb:.2f} GB")


def create_dataloaders(base_dir: str, tasks: List[str], 
                       batch_size: int = 4, num_workers: int = 4,
                       val_split: float = 0.2, image_size: int = 1024,
                       balance_tasks: bool = True,
                       max_slices_per_volume: int = None,
                       fast_augment: bool = False,
                       cached_slice_dir: Optional[str] = None,
                       split_mode: str = 'patient',
                       split_seed: int = 42,
                       num_folds: int = 1,
                       fold_index: int = 0):
    """
    Create train and validation dataloaders
    
    Args:
        base_dir: Base directory with all tasks
        tasks: List of task names
        batch_size: Batch size
        num_workers: Number of workers for data loading
        val_split: Validation split ratio
        image_size: Image size for SAM (1024 recommended)
        balance_tasks: Whether to oversample smaller tasks
        max_slices_per_volume: Optional cap of annotated slices read per volume
        fast_augment: Use lighter augmentation pipeline for faster training
        cached_slice_dir: Path to precomputed 2D slice cache
        split_mode: 'patient' (recommended) or 'slice'
        split_seed: Random seed for reproducible splitting
        num_folds: If >1, use patient-level k-fold cross validation
        fold_index: Which validation fold to use when num_folds > 1
    
    Returns:
        train_loader, val_loader, split_info
    """
    
    # Load full dataset without oversampling so split is leakage-safe.
    full_dataset = MedicalDecathlonDataset(
        base_dir=base_dir,
        tasks=tasks,
        split='training',
        transform=None,
        balance_tasks=False,
        max_slices_per_volume=max_slices_per_volume,
        cached_slice_dir=cached_slice_dir
    )

    all_samples = [dict(sample) for sample in full_dataset.samples]

    if num_folds > 1:
        train_samples, val_samples, split_info = patient_level_kfold_split(
            all_samples,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=split_seed
        )
    elif val_split <= 0:
        train_samples = all_samples
        val_samples = []
        split_info = {
            'split_mode': f'{split_mode}_full_train',
            'seed': split_seed,
            'val_ratio': 0.0,
            'num_folds': 1,
            'fold_index': 0,
            'n_total_volumes': None,
            'n_train_volumes': None,
            'n_val_volumes': 0,
            'n_train_slices': len(train_samples),
            'n_val_slices': 0,
        }
    elif split_mode == 'patient':
        train_samples, val_samples, split_info = patient_level_holdout_split(
            all_samples,
            val_ratio=val_split,
            seed=split_seed
        )
    elif split_mode == 'slice':
        if not 0.0 < val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")

        rng = random.Random(split_seed)
        rng.shuffle(all_samples)
        n_val = max(1, int(round(len(all_samples) * val_split)))
        n_val = min(n_val, len(all_samples) - 1)
        val_samples = all_samples[:n_val]
        train_samples = all_samples[n_val:]
        split_info = {
            'split_mode': 'slice',
            'seed': split_seed,
            'val_ratio': val_split,
            'num_folds': 1,
            'fold_index': 0,
            'n_total_volumes': None,
            'n_train_volumes': None,
            'n_val_volumes': None,
            'n_train_slices': len(train_samples),
            'n_val_slices': len(val_samples),
        }
    else:
        raise ValueError("split_mode must be 'patient' or 'slice'")

    train_dataset = MedicalDecathlonSubset(
        full_dataset,
        train_samples,
        transform=get_train_transforms(image_size, fast_mode=fast_augment),
        balance_tasks=balance_tasks
    )
    val_dataset = MedicalDecathlonSubset(
        full_dataset,
        val_samples,
        transform=get_val_transforms(image_size),
        balance_tasks=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    print(f"\nDataset split:")
    if split_info.get('n_train_volumes') is not None:
        print(f"  Volumes: {split_info['n_train_volumes']} train / {split_info['n_val_volumes']} val")
    print(f"  Slices: {split_info['n_train_slices']} train / {split_info['n_val_slices']} val")
    print(f"  Split mode: {split_info['split_mode']}")
    if split_info.get('num_folds', 1) > 1:
        print(f"  Fold: {split_info['fold_index'] + 1}/{split_info['num_folds']}")
    
    return train_loader, val_loader, split_info
