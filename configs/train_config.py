"""
Training Configuration for MedEViT-SAM Multi-Task Medical Segmentation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for training MedEViT-SAM on Medical Segmentation Decathlon (MSD)"""
    
    # Data Paths & Settings
    base_dir: str = './data/msd'
    tasks: Optional[List[str]] = None
    image_size: int = 512  # MedEViT-SAM native input size
    val_split: float = 0.2
    balance_tasks: bool = True
    split_mode: str = 'patient'
    split_seed: int = 42
    num_folds: int = 1
    fold_index: int = 0
    
    # Model Architecture
    model_name: str = 'efficientvit-sam-l0'  # Base architecture variant
    checkpoint_path: Optional[str] = None    # Path to pretrained checkpoint
    freeze_image_encoder: bool = False      # Fine-tune image encoder
    freeze_prompt_encoder: bool = True     # Freeze prompt encoder
    
    # Training Hyperparameters (Paper Section IV.C)
    epochs: int = 35
    warmup_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6      # Floor for cosine decay
    
    # Compound Loss Weights (Paper Section III.D)
    # L_total = w_t * L_Tversky + w_f * L_Focal + w_b * L_Boundary
    dice_weight: float = 1.0       # Tversky loss weight (w_t = 1.0)
    focal_weight: float = 0.5      # Focal loss weight (w_f = 0.5)
    boundary_weight: float = 0.5   # Boundary loss weight (w_b = 0.5)
    
    # Optimization
    optimizer: str = 'adamw'
    lr_scheduler: str = 'cosine'
    gradient_clip: float = 1.0
    
    # Hardware & Performance
    device: str = 'cuda'
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_interval: int = 5
    checkpoint_percent: float = 0.0
    checkpoint_epoch_percent: float = 0.0
    save_best: bool = True
    metric_for_best: str = 'dice'
    
    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = 'medevitsam-segmentation'
    wandb_run_name: Optional[str] = None
    eval_interval: int = 1
    
    def __post_init__(self):
        """Set defaults after initialization"""
        if self.tasks is None:
            self.tasks = [
                'Task01_BrainTumour',
                'Task02_Heart',
                'Task03_Liver',
                'Task04_Hippocampus',
                'Task05_Prostate',
                'Task06_Lung',
                'Task07_Pancreas',
                'Task08_HepaticVessel',
                'Task09_Spleen',
                'Task10_Colon'
            ]
        
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'base_dir': self.base_dir,
            'tasks': self.tasks,
            'image_size': self.image_size,
            'val_split': self.val_split,
            'balance_tasks': self.balance_tasks,
            'split_mode': self.split_mode,
            'split_seed': self.split_seed,
            'model_name': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'freeze_image_encoder': self.freeze_image_encoder,
            'freeze_prompt_encoder': self.freeze_prompt_encoder,
        }


def get_standard_config() -> TrainingConfig:
    """Standard training configuration across all 10 MSD tasks"""
    return TrainingConfig(
        model_name='efficientvit-sam-l0',
        epochs=35,
        warmup_epochs=5,
        batch_size=16,
        learning_rate=5e-5,
        freeze_image_encoder=False,
        freeze_prompt_encoder=True,
        save_interval=5,
        eval_interval=1
    )


def get_quick_test_config() -> TrainingConfig:
    """Quick test configuration for code verification"""
    return TrainingConfig(
        model_name='efficientvit-sam-l0',
        epochs=2,
        warmup_epochs=1,
        batch_size=2,
        save_interval=1,
        eval_interval=1
    )
