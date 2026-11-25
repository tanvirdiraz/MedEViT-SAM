"""
Custom Loss Functions for Medical Image Segmentation

Combines multiple loss functions optimized for medical imaging:
- Tversky Loss (Replaces standard Dice for severe class imbalance)
- Focal Loss (Handles confidence imbalance)
- Boundary Loss (Improves edge accuracy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class TverskyLoss(nn.Module):
    """
    Tversky Loss for highly imbalanced segmentation tasks (like blood vessels).
    alpha: weight for False Positives (guessing vessel where there is none)
    beta: weight for False Negatives (missing a real vessel)
    Setting beta > alpha forces the model to hunt for thin structures.
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - predicted masks (logits or probs)
            target: (B, 1, H, W) - ground truth masks
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Calculate True Positives, False Positives, and False Negatives
        TP = (pred * target).sum(dim=1)
        FP = (pred * (1 - target)).sum(dim=1)
        FN = ((1 - pred) * target).sum(dim=1)
        
        # Tversky Index Formula
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    alpha > 0.5 upweights the rare foreground class, which is correct for
    medical segmentation where positives (lesions, vessels, etc.) are far
    less frequent than background voxels.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss for improving edge accuracy"""
    
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Compute gradients (edges)
        pred_dx = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dy = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # L1 loss on boundaries
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        
        return (loss_x + loss_y) / 2


class CombinedLoss(nn.Module):
    """Combined loss for medical segmentation using Tversky"""
    
    def __init__(self, dice_weight=1.0, focal_weight=0.5, boundary_weight=0.5):
        super(CombinedLoss, self).__init__()
        # TverskyLoss (replaces DiceLoss) — FN-heavy penalty for rare structures
        self.base_loss = TverskyLoss(alpha=0.3, beta=0.7)
        # FocalLoss with corrected alpha=0.75 to upweight rare foreground class
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.boundary_loss = BoundaryLoss()
        
        # Kept the variable name 'dice_weight' to prevent breaking train_config.py
        self.base_weight = dice_weight 
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, pred, target):
        base = self.base_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.base_weight * base + 
                     self.focal_weight * focal + 
                     self.boundary_weight * boundary)
        
        return total_loss, {
            'dice': base.item(),  # Logged as 'dice' so train.py doesn't crash
            'focal': focal.item(),
            'boundary': boundary.item(),
            'total': total_loss.item()
        }


def compute_dice_score(pred, target, threshold=0.5):
    """Compute Dice score for evaluation (No .item() to prevent RAM/eval crashes)"""
    pred = torch.sigmoid(pred) if pred.min() < 0 else pred
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    dice = (2. * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)
    
    return float(dice)


def compute_iou_score(pred, target, threshold=0.5):
    """Compute IoU score for evaluation (No .item() to prevent RAM/eval crashes)"""
    pred = torch.sigmoid(pred) if pred.min() < 0 else pred
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return float(iou)


def compute_nsd_score(pred, target, threshold=0.5, tolerance=2.0):
    """
    Compute Normalized Surface Distance (NSD)
    Tolerance is in pixels (default 2.0 represents an acceptable boundary error).
    Requires scipy.ndimage.distance_transform_edt.
    """
    pred = torch.sigmoid(pred) if pred.min() < 0 else pred
    pred_binary = (pred > threshold).cpu().numpy().astype(bool)
    target_binary = target.cpu().numpy().astype(bool)
    
    batch_size = pred_binary.shape[0]
    nsd_scores = []
    
    for b in range(batch_size):
        p = pred_binary[b, 0]
        t = target_binary[b, 0]
        
        # If both are empty, perfect match
        if not p.any() and not t.any():
            nsd_scores.append(1.0)
            continue
            
        # If one is empty but the other is not, zero score
        if not p.any() or not t.any():
            nsd_scores.append(0.0)
            continue
            
        # Extract boundaries (pixels where neighbors change)
        import scipy.ndimage as ndimage
        struct = ndimage.generate_binary_structure(2, 1)
        p_boundary = p ^ ndimage.binary_erosion(p, structure=struct)
        t_boundary = t ^ ndimage.binary_erosion(t, structure=struct)
        
        if not p_boundary.any() or not t_boundary.any():
            nsd_scores.append(0.0)
            continue
            
        # Distance transforms
        dist_to_t = ndimage.distance_transform_edt(~t_boundary)
        dist_to_p = ndimage.distance_transform_edt(~p_boundary)
        
        # Surface distances
        surf_dist_p_to_t = dist_to_t[p_boundary]
        surf_dist_t_to_p = dist_to_p[t_boundary]
        
        # Calculate valid boundary pixels within tolerance
        valid_p = (surf_dist_p_to_t <= tolerance).sum()
        valid_t = (surf_dist_t_to_p <= tolerance).sum()
        
        total_p = p_boundary.sum()
        total_t = t_boundary.sum()
        
        # NSD formula
        nsd = (valid_p + valid_t) / (total_p + total_t + 1e-8)
        nsd_scores.append(nsd)
        
    return float(np.mean(nsd_scores))