"""
Multi-Task Fine-Tuning for EfficientViT-SAM on Medical Decathlon

Trains a unified model across all 5 Medical Decathlon tasks to achieve >0.82 Dice score
without task-specific fine-tuning.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# EfficientViT-SAM imports
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

# Modular imports
try:
    from configs.train_config import TrainingConfig, get_standard_config
    from data.dataset import create_dataloaders, get_volume_cache_stats, configure_volume_cache
    from models.losses import CombinedLoss, compute_dice_score, compute_iou_score, compute_nsd_score
except ImportError:
    from train_config import TrainingConfig, get_standard_config
    from medical_dataset import create_dataloaders, get_volume_cache_stats, configure_volume_cache
    from losses import CombinedLoss, compute_dice_score, compute_iou_score, compute_nsd_score


class MedicalSAMFineTuner:
    """Fine-tuner for EfficientViT-SAM on medical images"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize fine-tuner
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize loss
        self.criterion = CombinedLoss(
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            boundary_weight=config.boundary_weight
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = None
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if (config.mixed_precision and self.device.type == 'cuda') else None
        
        # Training state
        self.current_epoch = 0
        self.current_batch_idx = -1
        self.global_step = 0
        self.best_metric = 0.0
        self.train_history = []
        self.val_history = []
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._init_wandb()
    
    def _load_model(self):
        """Load EfficientViT-SAM model"""
        print(f"\nLoading model: {self.config.model_name}")
        
        # Load pretrained model
        model = create_efficientvit_sam_model(
            name=self.config.model_name,
            pretrained=True,
            weight_url=None
        )
        
        model = model.to(self.device)
        
        # Freeze components if specified
        if self.config.freeze_image_encoder:
            print("Freezing image encoder")
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        
        if self.config.freeze_prompt_encoder:
            print("Freezing prompt encoder")
            for param in model.prompt_encoder.parameters():
                param.requires_grad = False
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=1e-6
            )
        elif self.config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_training_steps // 3,
                gamma=0.1
            )
        elif self.config.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict()
            )
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.config.use_wandb = False

    def _forward_batch(self, images, bboxes, masks):
        """Run SAM forward with batched image encoder and per-image decode.

        SAM mask decoder natively supports many prompts for a single image. If we
        pass B image embeddings and B prompts together, it can expand internally
        to B*B outputs. To keep correct 1:1 image-prompt mapping, decode per-image.
        """
        image_embeddings_batch = self.model.image_encoder(images)

        batch_size = images.shape[0]
        pred_masks_list = []
        iou_predictions_list = []

        for i in range(batch_size):
            image_embeddings = image_embeddings_batch[i:i+1]
            bbox_i = bboxes[i:i+1]

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox_i,
                masks=None
            )

            pred_mask, iou_pred = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            pred_masks_list.append(pred_mask)
            iou_predictions_list.append(iou_pred)

        pred_masks = torch.cat(pred_masks_list, dim=0)
        iou_predictions = torch.cat(iou_predictions_list, dim=0)

        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.float(),
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return pred_masks, iou_predictions
    
    def train_epoch(self, train_loader, epoch, start_batch_idx=0,
                    checkpoint_steps_interval=None, total_steps=None,
                    checkpoint_epoch_steps_interval=None):
        """Train for one epoch"""
        self.model.train()
        
        # Compute epoch-level LR: linear warmup → cosine decay
        min_lr = getattr(self.config, 'min_lr', 1e-6)
        if epoch < self.config.warmup_epochs:
            # Linear warmup from warmup_lr to learning_rate
            warmup_progress = (epoch + 1) / max(1, self.config.warmup_epochs)
            lr = self.config.warmup_lr + warmup_progress * (self.config.learning_rate - self.config.warmup_lr)
        else:
            # Cosine decay from learning_rate down to min_lr
            decay_epochs = max(1, self.config.epochs - self.config.warmup_epochs)
            progress = (epoch - self.config.warmup_epochs) / decay_epochs
            lr = min_lr + 0.5 * (self.config.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_nsd = 0
        processed_batches = 0
        loss_components = {'dice': 0, 'focal': 0, 'boundary': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        epoch_total_batches = max(1, len(train_loader) - start_batch_idx)
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx < start_batch_idx:
                continue

            self.current_batch_idx = batch_idx
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            bboxes = batch['bbox'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    pred_masks, iou_predictions = self._forward_batch(images, bboxes, masks)
                    
                    # Compute loss
                    loss, loss_dict = self.criterion(pred_masks, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Regular training without mixed precision
                pred_masks, iou_predictions = self._forward_batch(images, bboxes, masks)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_masks, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                dice = compute_dice_score(pred_masks, masks)
                iou = compute_iou_score(pred_masks, masks)
                nsd = compute_nsd_score(pred_masks, masks)
            
            # Update statistics
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_nsd += nsd
            processed_batches += 1
            for key in loss_dict:
                loss_components[key] += loss_dict[key]

            self.global_step += 1

            if checkpoint_steps_interval and (self.global_step % checkpoint_steps_interval == 0):
                effective_total_steps = max(1, total_steps or self.global_step)
                progress_pct = min(100.0, 100.0 * self.global_step / effective_total_steps)
                progress_label = min(100, max(1, int(round(progress_pct))))
                checkpoint_name = (
                    f"checkpoint_progress_{progress_label:03d}pct_"
                    f"step_{self.global_step}.pth"
                )
                self.save_checkpoint(
                    checkpoint_name,
                    intra_epoch=True,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    progress_pct=progress_pct
                )

            if (
                checkpoint_epoch_steps_interval
                and (processed_batches % checkpoint_epoch_steps_interval == 0)
                and (processed_batches < epoch_total_batches)
            ):
                epoch_progress_pct = min(100.0, 100.0 * processed_batches / epoch_total_batches)
                epoch_progress_label = min(100, max(1, int(round(epoch_progress_pct))))
                checkpoint_name = (
                    f"checkpoint_epoch{epoch+1:03d}_progress_{epoch_progress_label:03d}pct_"
                    f"step_{self.global_step}.pth"
                )
                self.save_checkpoint(
                    checkpoint_name,
                    intra_epoch=True,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    progress_pct=epoch_progress_pct
                )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice:.4f}",
                'nsd': f"{nsd:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb
            if self.config.use_wandb and (batch_idx % self.config.log_interval == 0):
                import wandb
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_dice': dice,
                    'train/batch_iou': iou,
                    'train/batch_nsd': nsd,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Average metrics
        n_batches = max(1, processed_batches)
        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches
        avg_iou = total_iou / n_batches
        avg_nsd = total_nsd / n_batches
        avg_loss_components = {k: v / n_batches for k, v in loss_components.items()}

        self.current_batch_idx = -1
        
        # Print cache statistics
        cache_stats = get_volume_cache_stats()
        print(f"\nCache Stats - Hit Rate: {cache_stats['hit_rate']:.1f}% ({cache_stats['hits']}/{cache_stats['hits']+cache_stats['misses']}) | "
              f"Cached Volumes: {cache_stats['cached_volumes']} | "
              f"Memory: {cache_stats['current_size_gb']:.2f}GB / {cache_stats['max_size_gb']:.1f}GB")
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou,
            'nsd': avg_nsd,
            'loss_components': avg_loss_components
        }
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_nsd = 0
        task_metrics = {task: {'dice': [], 'iou': [], 'nsd': []} for task in self.config.tasks}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                bboxes = batch['bbox'].to(self.device, non_blocking=True)
                task_names = batch['task_name']

                pred_masks, iou_predictions = self._forward_batch(images, bboxes, masks)
                
                # Compute loss
                loss, _ = self.criterion(pred_masks, masks)
                
                # Compute metrics
                dice = compute_dice_score(pred_masks, masks)
                iou = compute_iou_score(pred_masks, masks)
                nsd = compute_nsd_score(pred_masks, masks)
                
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                total_nsd += nsd
                
                # Per-task metrics
                for i, task_name in enumerate(task_names):
                    task_dice = compute_dice_score(
                        pred_masks[i:i+1],
                        masks[i:i+1]
                    )
                    task_iou = compute_iou_score(
                        pred_masks[i:i+1],
                        masks[i:i+1]
                    )
                    task_nsd = compute_nsd_score(
                        pred_masks[i:i+1],
                        masks[i:i+1]
                    )
                    task_metrics[task_name]['dice'].append(task_dice)
                    task_metrics[task_name]['iou'].append(task_iou)
                    task_metrics[task_name]['nsd'].append(task_nsd)
        
        # Average metrics
        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches
        avg_iou = total_iou / n_batches
        avg_nsd = total_nsd / n_batches
        
        # Per-task averages
        task_avg = {}
        for task, metrics in task_metrics.items():
            if metrics['dice']:
                task_avg[task] = {
                    'dice': np.mean(metrics['dice']),
                    'iou': np.mean(metrics['iou']),
                    'nsd': np.mean(metrics['nsd'])
                }
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou,
            'nsd': avg_nsd,
            'task_metrics': task_avg
        }
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        has_validation = val_loader is not None and len(val_loader) > 0

        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Configuration: {self.config.model_name}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Tasks: {', '.join(self.config.tasks)}")
        print(f"Validation enabled: {'yes' if has_validation else 'no (final full-data retrain mode)'}")
        print("="*60 + "\n")
        
        # Create scheduler
        num_training_steps = self.config.epochs * len(train_loader)
        self.scheduler = self._create_scheduler(num_training_steps)

        checkpoint_steps_interval = None
        checkpoint_percent = float(getattr(self.config, "checkpoint_percent", 0.0) or 0.0)
        if checkpoint_percent > 0:
            if checkpoint_percent > 100:
                raise ValueError("checkpoint_percent must be in (0, 100].")
            checkpoint_steps_interval = max(1, int(num_training_steps * (checkpoint_percent / 100.0)))
            print(
                f"Percent checkpoints enabled: every {checkpoint_percent:.2f}% "
                f"(~{checkpoint_steps_interval} steps)."
            )

        checkpoint_epoch_steps_interval = None
        checkpoint_epoch_percent = float(getattr(self.config, "checkpoint_epoch_percent", 0.0) or 0.0)
        if checkpoint_epoch_percent > 0:
            if checkpoint_epoch_percent > 100:
                raise ValueError("checkpoint_epoch_percent must be in (0, 100].")
            checkpoint_epoch_steps_interval = max(
                1,
                int(len(train_loader) * (checkpoint_epoch_percent / 100.0))
            )
            print(
                f"Per-epoch checkpoints enabled: every {checkpoint_epoch_percent:.2f}% "
                f"within epoch (~{checkpoint_epoch_steps_interval} steps)."
            )

        start_epoch = 0
        start_batch_idx = 0
        if getattr(self.config, "resume_checkpoint", None):
            start_epoch, start_batch_idx, resumed_global_step = self.resume_from_checkpoint(
                self.config.resume_checkpoint
            )
            self.global_step = resumed_global_step
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
            
            # Train
            train_metrics = self.train_epoch(
                train_loader,
                epoch,
                start_batch_idx=epoch_start_batch_idx,
                checkpoint_steps_interval=checkpoint_steps_interval,
                total_steps=num_training_steps,
                checkpoint_epoch_steps_interval=checkpoint_epoch_steps_interval
            )
            self.train_history.append(train_metrics)
            start_batch_idx = 0
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Dice: {train_metrics['dice']:.4f}")
            print(f"  Train IoU: {train_metrics['iou']:.4f}")
            print(f"  Train NSD: {train_metrics['nsd']:.4f}")
            
            # Validate
            if has_validation and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                self.val_history.append(val_metrics)
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Dice: {val_metrics['dice']:.4f}")
                print(f"  Val IoU: {val_metrics['iou']:.4f}")
                print(f"  Val NSD: {val_metrics['nsd']:.4f}")
                
                # Per-task metrics
                print("\n  Per-task metrics:")
                for task, metrics in val_metrics['task_metrics'].items():
                    print(f"    {task}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, NSD={metrics['nsd']:.4f}")
                
                # Log to wandb
                if self.config.use_wandb:
                    import wandb
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/loss': train_metrics['loss'],
                        'train/dice': train_metrics['dice'],
                        'train/iou': train_metrics['iou'],
                        'train/nsd': train_metrics['nsd'],
                        'val/loss': val_metrics['loss'],
                        'val/dice': val_metrics['dice'],
                        'val/iou': val_metrics['iou'],
                        'val/nsd': val_metrics['nsd']
                    }
                    # Add per-task metrics
                    for task, metrics in val_metrics['task_metrics'].items():
                        log_dict[f'val/{task}/dice'] = metrics['dice']
                        log_dict[f'val/{task}/iou'] = metrics['iou']
                        log_dict[f'val/{task}/nsd'] = metrics['nsd']
                    wandb.log(log_dict)
                
                # Save best model
                if self.config.save_best:
                    current_metric = val_metrics[self.config.metric_for_best]
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.save_checkpoint(f'best_model_{self.config.metric_for_best}.pth')
                        print(f"  ✓ Saved best model (Dice: {current_metric:.4f})")
            elif not has_validation and (epoch + 1) == 1:
                print("  Skipping validation: training on all labeled data for final model export.")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric_value = val_metrics['dice'] if has_validation else train_metrics['dice']
                    self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()
            
            print()
        
        total_time = time.time() - start_time
        print("="*60)
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best {self.config.metric_for_best}: {self.best_metric:.4f}")
        print("="*60)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        self.save_training_history()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training state from a saved checkpoint.

        Returns:
            tuple[int, int, int]:
                next epoch index, next batch index, restored global step.
        """
        checkpoint_path = str(checkpoint_path)
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: could not load scheduler state ({e}). Continuing with a fresh scheduler.")

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                print(f"Warning: could not load AMP scaler state ({e}).")

        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        self.train_history = checkpoint.get("train_history", self.train_history)
        self.val_history = checkpoint.get("val_history", self.val_history)
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.current_batch_idx = int(checkpoint.get("batch_idx", -1))
        restored_global_step = int(checkpoint.get("global_step", 0))

        if bool(checkpoint.get("intra_epoch", False)) and self.current_batch_idx >= 0:
            next_epoch = self.current_epoch
            next_batch_idx = self.current_batch_idx + 1
            print(
                f"Loaded intra-epoch checkpoint at epoch={self.current_epoch}, "
                f"batch={self.current_batch_idx}. Resuming from batch {next_batch_idx}."
            )
        else:
            next_epoch = self.current_epoch + 1
            next_batch_idx = 0
            print(f"Loaded checkpoint epoch={self.current_epoch}. Next epoch will be {next_epoch}.")

        print(f"Best metric so far: {self.best_metric:.4f}")
        return next_epoch, next_batch_idx, restored_global_step

    def load_weights_only(self, checkpoint_path: str):
        """Load only model weights from checkpoint for phased transfer learning.

        This initializes model parameters from a previous phase without restoring
        optimizer/scheduler/epoch state, so training starts fresh at epoch 0.
        """
        checkpoint_path = str(checkpoint_path)
        print(f"\nLoading model weights only from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid checkpoint format (missing model_state_dict): {checkpoint_path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully. Starting new phase from epoch 0.")
    
    def save_checkpoint(self, filename, intra_epoch=False, batch_idx=None,
                        global_step=None, progress_pct=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'batch_idx': self.current_batch_idx if batch_idx is None else int(batch_idx),
            'intra_epoch': bool(intra_epoch),
            'global_step': self.global_step if global_step is None else int(global_step),
            'progress_pct': progress_pct,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")
    
    def save_training_history(self):
        """Save training history as JSON"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config.to_dict()
        }
        
        save_path = self.save_dir / 'training_history.json'
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {save_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune EfficientViT-SAM on Medical Decathlon')
    parser.add_argument('--config', type=str, default='standard',
                       choices=['quick', 'standard', 'high_quality', 'fast', 'hippocampus', 'tri_tasks'],
                       help='Training configuration preset')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Input image size (overrides config)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='DataLoader workers (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Save directory (overrides config)')
    parser.add_argument('--save_interval', type=int, default=None,
                       help='Save checkpoint every N epochs (overrides config)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')

    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory containing TaskXX_* folders (overrides config)')
    parser.add_argument('--tasks', type=str, default=None,
                       help='Comma-separated task folder names, e.g. Task01_BrainTumour,Task02_Heart,... (overrides config)')
    parser.add_argument('--no_balance', action='store_true',
                       help='Disable task balancing/oversampling (recommended for many tasks)')
    parser.add_argument('--split_mode', type=str, default=None,
                       choices=['patient', 'slice'],
                       help='Validation split mode (patient recommended for reliable medical evaluation)')
    parser.add_argument('--val_split', type=float, default=None,
                       help='Validation ratio for holdout split (overrides config)')
    parser.add_argument('--split_seed', type=int, default=None,
                       help='Random seed for reproducible train/validation split')
    parser.add_argument('--num_folds', type=int, default=None,
                       help='If >1, use patient-level k-fold cross-validation')
    parser.add_argument('--fold_index', type=int, default=None,
                       help='Validation fold index when using --num_folds')

    parser.add_argument('--resume', type=str, default=None,
                       help='Path to a checkpoint (.pth) to resume training from')
    parser.add_argument('--resume_weights_only', action='store_true',
                       help='Load only model weights from --resume and start a fresh phase at epoch 0')
    parser.add_argument('--min_lr', type=float, default=None,
                       help='Minimum LR for cosine decay floor (default: 1e-6)')
    parser.add_argument('--checkpoint_percent', type=float, default=None,
                       help='Save intra-epoch progress checkpoints every N%% of total training (e.g., 1.0)')
    parser.add_argument('--checkpoint_epoch_percent', type=float, default=None,
                       help='Save progress checkpoints every N%% within each epoch (e.g., 20 for 20/40/60/80%%)')
    parser.add_argument('--max_slices_per_volume', type=int, default=None,
                       help='Optional cap of annotated slices loaded per volume to reduce epoch size and RAM')
    parser.add_argument('--fast_aug', action='store_true',
                       help='Use lighter/faster training augmentations')
    parser.add_argument('--cache_gb', type=float, default=None,
                       help='Limit in-RAM volume cache size in GB (e.g., 1.5 or 2.0)')
    parser.add_argument('--slice_cache_dir', type=str, default=None,
                       help='Directory containing precomputed 2D slice cache (slice_index.json + .npz files)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'quick':
        from train_config import get_quick_test_config
        config = get_quick_test_config()
    elif args.config == 'high_quality':
        from train_config import get_high_quality_config
        config = get_high_quality_config()
    elif args.config == 'fast':
        from train_config import get_fast_convergence_config
        config = get_fast_convergence_config()
    elif args.config == 'hippocampus':
        from train_config import get_hippocampus_config
        config = get_hippocampus_config()
    elif args.config == 'tri_tasks':
        from train_config import get_tri_task_config
        config = get_tri_task_config()
    else:
        config = get_standard_config()
    
    # Override config with command-line arguments
    if args.model:
        config.model_name = args.model
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.image_size:
        config.image_size = args.image_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.lr:
        config.learning_rate = args.lr
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.save_interval is not None:
        if args.save_interval < 1:
            raise ValueError("save_interval must be >= 1")
        config.save_interval = args.save_interval
    if args.use_wandb:
        config.use_wandb = True

    if args.base_dir:
        config.base_dir = args.base_dir
    if args.tasks:
        config.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    if args.no_balance:
        config.balance_tasks = False
    if args.split_mode:
        config.split_mode = args.split_mode
    if args.val_split is not None:
        config.val_split = args.val_split
    if args.split_seed is not None:
        config.split_seed = args.split_seed
    if args.num_folds is not None:
        config.num_folds = args.num_folds
    if args.fold_index is not None:
        config.fold_index = args.fold_index

    if args.resume and not args.resume_weights_only:
        # Keep on config so MedicalSAMFineTuner.train() can automatically resume.
        config.resume_checkpoint = args.resume
    if args.min_lr is not None:
        config.min_lr = args.min_lr
    if args.checkpoint_percent is not None:
        config.checkpoint_percent = args.checkpoint_percent
    if args.checkpoint_epoch_percent is not None:
        config.checkpoint_epoch_percent = args.checkpoint_epoch_percent

    if args.cache_gb is not None:
        configure_volume_cache(args.cache_gb)

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, split_info = create_dataloaders(
        base_dir=config.base_dir,
        tasks=config.tasks,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        image_size=config.image_size,
        balance_tasks=config.balance_tasks,
        max_slices_per_volume=args.max_slices_per_volume,
        fast_augment=args.fast_aug,
        cached_slice_dir=args.slice_cache_dir,
        split_mode=config.split_mode,
        split_seed=config.split_seed,
        num_folds=config.num_folds,
        fold_index=config.fold_index
    )

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    split_summary_path = Path(config.save_dir) / 'data_split_summary.json'
    split_summary = {
        'config': config.to_dict(),
        'split_info': split_info,
    }
    with open(split_summary_path, 'w', encoding='utf-8') as f:
        json.dump(split_summary, f, indent=2)
    print(f"Data split summary saved: {split_summary_path}")
    
    # Create fine-tuner and train
    fine_tuner = MedicalSAMFineTuner(config)

    if args.resume and args.resume_weights_only:
        fine_tuner.load_weights_only(args.resume)

    fine_tuner.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
