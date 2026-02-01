#!/usr/bin/env python3
"""
Sparse Gate-Guided Gaussian Training Script

Fully leverages the TOPS-Gate by:
1. Initializing Gaussians only in gated regions
2. Computing forward pass only at gated voxels
3. Computing loss only on gated voxels
4. ~10-20x speedup for sparse volumes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tifffile
import math
import os
import sys
import json
import argparse
from tqdm import tqdm
from dataclasses import dataclass

# Try to import fused_ssim and lpips
try:
    from fused_ssim import fused_ssim
    HAS_FUSED_SSIM = True
except ImportError:
    HAS_FUSED_SSIM = False
    print("Warning: fused_ssim not available, using pytorch_msssim")
    try:
        from pytorch_msssim import ssim as pytorch_ssim
        HAS_PYTORCH_SSIM = True
    except ImportError:
        HAS_PYTORCH_SSIM = False
        print("Warning: pytorch_msssim not available, SSIM will be disabled")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not available, LPIPS metric will be disabled")

# Add paths
sys.path.insert(0, '/mnt/intelpa-1/armin/containers/storage/Method2')
sys.path.insert(0, '/mnt/intelpa-1/armin/containers/storage/Method2/Dec24_GaussianVolume')

from gaussian_model_cuda import CUDAGaussianModel
from losses import (
    ReconstructionLoss, 
    SparsityRegularization, 
    OverlapRegularization, 
    SmoothnessRegularization,
    compute_psnr,
    compute_ssim
)
from scipy import ndimage


def compute_edge_weights(volume: np.ndarray, sigma: float = 1.0, base_weight: float = 1.0, edge_boost: float = 5.0) -> np.ndarray:
    """Compute edge-aware weights for loss - higher weight on dendrite boundaries.
    
    Args:
        volume: 3D volume (D, H, W)
        sigma: Gaussian blur sigma for gradient computation
        base_weight: Minimum weight for flat regions
        edge_boost: Additional weight multiplier for edges
    
    Returns:
        weights: Same shape as volume, higher values on edges
    """
    # Compute gradient magnitude using Sobel
    grad_d = ndimage.sobel(volume, axis=0)
    grad_h = ndimage.sobel(volume, axis=1)
    grad_w = ndimage.sobel(volume, axis=2)
    grad_mag = np.sqrt(grad_d**2 + grad_h**2 + grad_w**2)
    
    # Normalize to [0, 1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    
    # Smooth slightly to avoid noise
    if sigma > 0:
        grad_mag = ndimage.gaussian_filter(grad_mag, sigma=sigma)
    
    # Weight = base + edge_boost * gradient
    weights = base_weight + edge_boost * grad_mag
    
    return weights.astype(np.float32)


# ============== TOPS-Gate Model Definition ==============
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, depth=4, act=nn.SiLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth-1):
            layers += [nn.Linear(d, hidden_dim), act()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def positional_encoding(x, L=6):
    freqs = (2.0 ** torch.arange(L, device=x.device)).view(1, L, 1)
    x = x.unsqueeze(1)
    angles = 2*math.pi*freqs*x
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return pe.flatten(1)


class TOPSGate(nn.Module):
    def __init__(self, brick_size=16, code_dim=16, pe_L=6):
        super().__init__()
        self.brick_size = int(brick_size)
        self.code_dim = int(code_dim)
        self.pe_L = int(pe_L)
        self.occ_mlp = MLP(in_dim=3 + 6*pe_L, hidden_dim=256, out_dim=1, depth=5)
        self.int_mlp = MLP(in_dim=3 + 6*pe_L + code_dim, hidden_dim=256, out_dim=1, depth=6)
        self._code_init = nn.Parameter(torch.randn(code_dim) * 0.01)

    def occ_prob(self, x01):
        pe = positional_encoding(x01, L=self.pe_L)
        inp = torch.cat([x01, pe], dim=-1)
        logits = self.occ_mlp(inp).squeeze(-1)
        return torch.sigmoid(logits)


def load_tops_gate_model(checkpoint_path: str, device: str = 'cuda'):
    print(f"Loading TOPS-Gate from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict']
    code_dim = state['_code_init'].shape[0]
    occ_mlp_in = state['occ_mlp.net.0.weight'].shape[1]
    pe_L = (occ_mlp_in - 3) // 6
    print(f"  Detected config: code_dim={code_dim}, pe_L={pe_L}")
    model = TOPSGate(brick_size=16, code_dim=code_dim, pe_L=pe_L).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded from step {ckpt['step']}")
    return model


def generate_hard_gate_mask(gate_model: TOPSGate, volume_shape, tau: float = 0.5, device: str = 'cuda'):
    D, H, W = volume_shape
    total_voxels = D * H * W
    d = (torch.arange(D, device=device, dtype=torch.float32) + 0.5) / D
    h = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    w = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    mask = torch.zeros(D, H, W, device=device)
    print(f"Generating hard gate mask (tau={tau})...")
    with torch.no_grad():
        for d_idx in range(D):
            hh, ww = torch.meshgrid(h, w, indexing='ij')
            dd = torch.full_like(hh, d[d_idx].item())
            coords = torch.stack([dd.flatten(), hh.flatten(), ww.flatten()], dim=1)
            probs = gate_model.occ_prob(coords)
            mask[d_idx] = (probs > tau).float().view(H, W)
    occupancy = mask.sum().item() / total_voxels * 100
    print(f"  Gate occupancy: {occupancy:.2f}% ({int(mask.sum().item()):,} / {total_voxels:,} voxels)")
    return mask


@dataclass
class SparseTrainerConfig:
    learning_rate: float = 0.01
    lr_decay: float = 0.1  # Final LR = initial * lr_decay
    lr_decay_start: int = 2500  # Start decay after densification stops
    finetune_lr: float = 0.002  # Lower LR for fine-tuning phase
    lambda_sparsity: float = 0.001  # Sparsity regularization
    lambda_overlap: float = 0.001  # Overlap regularization
    lambda_smoothness: float = 0.001  # Smoothness regularization
    smoothness_neighbors: int = 5  # k-NN for smoothness
    knn_k: int = 32  # Neighbors for overlap computation
    # Densification thresholds (less aggressive - only top ~10% gradients trigger)
    densify_grad_threshold: float = 0.0001   # Higher = less aggressive (max observed ~8e-05)
    densify_scale_threshold: float = 0.001    # Clone small (<), split large (>)
    densify_interval: int = 500             # Less frequent (every 1000 epochs)
    densify_start: int = 500                # Start later for stable base
    densify_stop: int = 5000                 # Allow through most of training
    max_gaussians: int = 50000               # Cap on number of Gaussians
    # Pruning thresholds
    prune_intensity_threshold: float = 0.02  # Prune if sigmoid(intensity) < 3%
    prune_scale_threshold: float = 0.06      # Prune if max scale > 6% of volume
    # Edge-aware loss settings
    use_edge_weights: bool = False  # Disabled by default (not in proposal)
    edge_boost: float = 3.0  # How much to boost edge regions


class SparseGateGuidedTrainer:
    """
    Trainer that fully leverages gate for sparse computation.
    Only evaluates and computes loss on gated voxels.
    """
    
    def __init__(
        self,
        model: CUDAGaussianModel,
        volume: torch.Tensor,
        gate_mask: torch.Tensor,
        config: SparseTrainerConfig,
        device: str = 'cuda',
        output_dir: str = None
    ):
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = output_dir  # Output checkpoint folder
        
        # Volume
        self.volume = volume.to(device)
        self.volume_max = self.volume.max().item()
        if self.volume_max > 0:
            self.volume_norm = self.volume / self.volume_max
        else:
            self.volume_norm = self.volume
        
        self.D, self.H, self.W = self.volume.shape
        
        # Gate mask - SPARSE!
        self.gate_mask = gate_mask.bool().to(device)
        
        # Pre-compute gated voxel coordinates and values
        self.gated_indices = torch.nonzero(self.gate_mask, as_tuple=False)  # (N_gated, 3)
        self.n_gated = self.gated_indices.shape[0]
        
        # Normalized coordinates for gated voxels [0, 1]
        self.gated_coords = torch.zeros(self.n_gated, 3, device=device)
        self.gated_coords[:, 0] = (self.gated_indices[:, 0].float() + 0.5) / self.D
        self.gated_coords[:, 1] = (self.gated_indices[:, 1].float() + 0.5) / self.H
        self.gated_coords[:, 2] = (self.gated_indices[:, 2].float() + 0.5) / self.W
        
        # Ground truth values at gated voxels
        self.gated_gt = self.volume_norm[
            self.gated_indices[:, 0],
            self.gated_indices[:, 1],
            self.gated_indices[:, 2]
        ]
        
        # Edge-aware weights for loss (higher on dendrite boundaries)
        self.gated_weights = None
        if config.use_edge_weights:
            print("  Computing edge-aware loss weights...")
            edge_weights = compute_edge_weights(
                self.volume_norm.cpu().numpy(),
                sigma=1.0,
                base_weight=1.0,
                edge_boost=config.edge_boost
            )
            edge_weights_tensor = torch.from_numpy(edge_weights).to(device)
            self.gated_weights = edge_weights_tensor[
                self.gated_indices[:, 0],
                self.gated_indices[:, 1],
                self.gated_indices[:, 2]
            ]
            # Normalize weights to mean=1
            self.gated_weights = self.gated_weights / self.gated_weights.mean()
            print(f"    Weight range: [{self.gated_weights.min():.2f}, {self.gated_weights.max():.2f}]")
        
        print(f"  Sparse training on {self.n_gated:,} gated voxels ({100*self.n_gated/(self.D*self.H*self.W):.2f}%)")
        print(f"  Speedup: ~{(self.D*self.H*self.W)/self.n_gated:.1f}x vs dense")
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # History
        self.history = {
            'total_loss': [], 'mse_loss': [], 'sparsity_loss': [],
            'overlap_loss': [], 'smoothness_loss': [],
            'psnr': [], 'ssim': [], 'lpips': [], 'num_gaussians': [],
            'densify_split': [], 'densify_clone': [], 'densify_prune': [],
        }
        
        # Initialize loss modules from losses.py
        self.reconstruction_loss = ReconstructionLoss()
        self.sparsity_loss = SparsityRegularization(lambda_w=config.lambda_sparsity)
        self.use_overlap = config.lambda_overlap > 0
        self.use_smoothness = config.lambda_smoothness > 0
        if self.use_overlap:
            self.overlap_loss = OverlapRegularization(lambda_o=config.lambda_overlap)
        if self.use_smoothness:
            self.smoothness_loss = SmoothnessRegularization(
                lambda_s=config.lambda_smoothness,
                num_neighbors=config.smoothness_neighbors
            )
        
        # LPIPS model (lazy init)
        self._lpips_model = None
        
        # Gradient accumulator
        self.grad_accum = None
        self.grad_count = 0
    
    def compute_loss_sparse(self):
        """Compute loss ONLY on gated voxels using KNN - truly sparse!
        
        Uses modular loss functions from losses.py:
        - ReconstructionLoss (MSE) with optional edge-aware weighting
        - SparsityRegularization (L1 on weights)
        - OverlapRegularization (penalize overlapping Gaussians)
        - SmoothnessRegularization (kNN spatial coherence)
        """
        # Evaluate using K nearest Gaussians per query point
        # This is O(M*K) instead of O(M*N) - the REAL speedup!
        pred_values = self.model.forward_knn(self.gated_coords, k=self.config.knn_k, sigma_cutoff=5.0)
        
        # MSE only on gated voxels - with edge-aware weighting if enabled
        if self.gated_weights is not None:
            # Weighted MSE: higher weight on edge voxels (dendrite boundaries)
            mse_loss = (self.gated_weights * (pred_values - self.gated_gt) ** 2).mean()
        else:
            mse_loss = self.reconstruction_loss(pred_values, self.gated_gt)
        
        # Sparsity regularization: λ·Σ|wᵢ| (using losses.py)
        intensities = self.model.intensities()
        sparsity_loss = self.sparsity_loss(intensities)
        
        total_loss = mse_loss + sparsity_loss
        
        # Overlap regularization: λ·Σᵢ<ⱼ O(Gᵢ,Gⱼ) (using losses.py)
        overlap_loss = torch.tensor(0.0, device=self.device)
        if self.use_overlap:
            positions = self.model.positions
            covariance = self.model.get_covariance_matrices()
            overlap_loss = self.overlap_loss(positions, covariance)
            total_loss = total_loss + overlap_loss
        
        # Smoothness regularization: λ·Σᵢ Σⱼ∈𝒩(i) ||θᵢ-θⱼ||² (using losses.py)
        smoothness_loss = torch.tensor(0.0, device=self.device)
        if self.use_smoothness:
            positions = self.model.positions
            log_scales = self.model.log_scales()
            smoothness_loss = self.smoothness_loss(positions, intensities, log_scales)
            total_loss = total_loss + smoothness_loss
        
        return total_loss, mse_loss, sparsity_loss, overlap_loss, smoothness_loss
    
    def compute_loss_sparse_sampled(self, num_samples: int):
        """Compute loss on random subset of gated voxels (even faster).
        
        Uses modular loss functions from losses.py with edge-aware weighting.
        """
        if num_samples >= self.n_gated:
            return self.compute_loss_sparse()
        
        # Sample from gated voxels
        idx = torch.randint(0, self.n_gated, (num_samples,), device=self.device)
        coords = self.gated_coords[idx]
        gt_values = self.gated_gt[idx]
        
        pred_values = self.model.forward_knn(coords, k=self.config.knn_k, sigma_cutoff=5.0)
        
        # Weighted MSE if edge weights enabled
        if self.gated_weights is not None:
            weights = self.gated_weights[idx]
            mse_loss = (weights * (pred_values - gt_values) ** 2).mean()
        else:
            mse_loss = self.reconstruction_loss(pred_values, gt_values)
        
        intensities = self.model.intensities()
        sparsity_loss = self.sparsity_loss(intensities)
        
        total_loss = mse_loss + sparsity_loss
        
        # Overlap regularization (computed on all Gaussians, not sampled voxels)
        overlap_loss = torch.tensor(0.0, device=self.device)
        if self.use_overlap:
            positions = self.model.positions
            covariance = self.model.get_covariance_matrices()
            overlap_loss = self.overlap_loss(positions, covariance)
            total_loss = total_loss + overlap_loss
        
        # Smoothness regularization (computed on all Gaussians)
        smoothness_loss = torch.tensor(0.0, device=self.device)
        if self.use_smoothness:
            positions = self.model.positions
            log_scales = self.model.log_scales()
            smoothness_loss = self.smoothness_loss(positions, intensities, log_scales)
            total_loss = total_loss + smoothness_loss
        
        return total_loss, mse_loss, sparsity_loss, overlap_loss, smoothness_loss
    
    def compute_psnr_sparse(self, num_samples: int = 100000):
        """Compute PSNR on a sample of gated voxels using KNN."""
        with torch.no_grad():
            if num_samples >= self.n_gated:
                # Use all voxels (batched)
                pred = self.model.forward_knn(self.gated_coords, k=self.config.knn_k, sigma_cutoff=5.0)
                gt = self.gated_gt
            else:
                # Sample for faster evaluation
                idx = torch.randint(0, self.n_gated, (num_samples,), device=self.device)
                coords = self.gated_coords[idx]
                gt = self.gated_gt[idx]
                pred = self.model.forward_knn(coords, k=self.config.knn_k, sigma_cutoff=5.0)
            
            mse = F.mse_loss(pred, gt)
            if mse < 1e-10:
                return float('inf')
            return (10 * torch.log10(1.0 / mse)).item()
    
    def compute_ssim_mip(self):
        """Compute SSIM on MIP projections (XY, XZ, YZ)."""
        with torch.no_grad():
            pred_vol = self.model()  # (D, H, W)
            gt_vol = self.volume_norm
            
            ssim_vals = []
            
            # XY MIP (along Z/D axis)
            pred_xy = pred_vol.max(dim=0)[0].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gt_xy = gt_vol.max(dim=0)[0].unsqueeze(0).unsqueeze(0)
            
            # XZ MIP (along Y/H axis)
            pred_xz = pred_vol.max(dim=1)[0].unsqueeze(0).unsqueeze(0)  # (1, 1, D, W)
            gt_xz = gt_vol.max(dim=1)[0].unsqueeze(0).unsqueeze(0)
            
            # YZ MIP (along X/W axis)
            pred_yz = pred_vol.max(dim=2)[0].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H)
            gt_yz = gt_vol.max(dim=2)[0].unsqueeze(0).unsqueeze(0)
            
            for pred_mip, gt_mip in [(pred_xy, gt_xy), (pred_xz, gt_xz), (pred_yz, gt_yz)]:
                if HAS_FUSED_SSIM:
                    # fused_ssim expects (B, C, H, W) in [0, 1]
                    ssim_val = fused_ssim(pred_mip.clamp(0, 1), gt_mip.clamp(0, 1))
                elif HAS_PYTORCH_SSIM:
                    ssim_val = pytorch_ssim(pred_mip.clamp(0, 1), gt_mip.clamp(0, 1), data_range=1.0)
                else:
                    ssim_val = torch.tensor(0.0)
                ssim_vals.append(ssim_val.item())
            
            return np.mean(ssim_vals)
    
    def compute_lpips_mip(self):
        """Compute LPIPS on MIP projections."""
        if not HAS_LPIPS:
            return 0.0
        
        # Lazy init LPIPS model
        if self._lpips_model is None:
            self._lpips_model = lpips.LPIPS(net='vgg').to(self.device)
            self._lpips_model.eval()
        
        with torch.no_grad():
            pred_vol = self.model()
            gt_vol = self.volume_norm
            
            lpips_vals = []
            
            # XY MIP
            pred_xy = pred_vol.max(dim=0)[0]
            gt_xy = gt_vol.max(dim=0)[0]
            
            # XZ MIP
            pred_xz = pred_vol.max(dim=1)[0]
            gt_xz = gt_vol.max(dim=1)[0]
            
            # YZ MIP
            pred_yz = pred_vol.max(dim=2)[0]
            gt_yz = gt_vol.max(dim=2)[0]
            
            for pred_mip, gt_mip in [(pred_xy, gt_xy), (pred_xz, gt_xz), (pred_yz, gt_yz)]:
                # LPIPS expects (B, 3, H, W) in [-1, 1]
                # Repeat grayscale to RGB and scale to [-1, 1]
                pred_rgb = pred_mip.unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
                gt_rgb = gt_mip.unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
                
                # Resize if too small (LPIPS needs at least 64x64)
                min_size = 64
                if pred_rgb.shape[2] < min_size or pred_rgb.shape[3] < min_size:
                    pred_rgb = F.interpolate(pred_rgb, size=(max(min_size, pred_rgb.shape[2]), 
                                                              max(min_size, pred_rgb.shape[3])), 
                                            mode='bilinear', align_corners=False)
                    gt_rgb = F.interpolate(gt_rgb, size=(max(min_size, gt_rgb.shape[2]), 
                                                          max(min_size, gt_rgb.shape[3])), 
                                          mode='bilinear', align_corners=False)
                
                lpips_val = self._lpips_model(pred_rgb.clamp(-1, 1), gt_rgb.clamp(-1, 1))
                lpips_vals.append(lpips_val.item())
            
            return np.mean(lpips_vals)
    
    def densify_step(self, iteration: int):
        cfg = self.config
        if iteration < cfg.densify_start or iteration > cfg.densify_stop:
            return 0, 0, 0
        if iteration % cfg.densify_interval != 0:
            return 0, 0, 0
        if self.grad_accum is None or self.grad_count == 0:
            return 0, 0, 0
        
        avg_grads = self.grad_accum / self.grad_count
        N_before = self.model.N
        
        # Debug: print gradient stats at densify steps
        grad_max = avg_grads.max().item()
        grad_mean = avg_grads.mean().item()
        n_above_thresh = (avg_grads > cfg.densify_grad_threshold).sum().item()
        print(f"\n[Densify @ {iteration}] grad max={grad_max:.2e}, mean={grad_mean:.2e}, ")
        print(f"                   above_thresh={n_above_thresh}/{len(avg_grads)}, thresh={cfg.densify_grad_threshold:.2e}")
        
        n_clone = 0
        n_split = 0
        
        # Clone small Gaussians with high gradients (need more coverage)
        if self.model.N < cfg.max_gaussians:
            n_clone = self.model.densify_and_clone(
                avg_grads, cfg.densify_grad_threshold, cfg.densify_scale_threshold
            )
        
        # Split large Gaussians with high gradients (need finer detail)
        # Re-fetch gradients after clone since model size may have changed
        if self.model.N < cfg.max_gaussians:
            # Pad gradients if clone added Gaussians
            if len(avg_grads) < self.model.N:
                avg_grads = torch.cat([avg_grads, torch.zeros(self.model.N - len(avg_grads), device=avg_grads.device)])
            elif len(avg_grads) > self.model.N:
                avg_grads = avg_grads[:self.model.N]
            n_split = self.model.densify_and_split(
                avg_grads, cfg.densify_grad_threshold, cfg.densify_scale_threshold
            )
        
        # Prune low-contribution or excessively large Gaussians
        n_prune = self.model.prune_gaussians(
            intensity_threshold=cfg.prune_intensity_threshold,
            scale_threshold=cfg.prune_scale_threshold
        )
        
        print(f"                   clone={n_clone}, split={n_split}, prune={n_prune}, N: {N_before} -> {self.model.N}")
        
        if n_clone > 0 or n_split > 0 or n_prune > 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
            self.grad_accum = None
            self.grad_count = 0
        
        return n_split, n_clone, n_prune
    
    def accumulate_gradients(self):
        if self.model.positions_raw.grad is not None:
            grad_norm = self.model.positions_raw.grad.norm(dim=1)
            if self.grad_accum is None:
                self.grad_accum = grad_norm.clone()
            else:
                if self.grad_accum.shape[0] != grad_norm.shape[0]:
                    self.grad_accum = grad_norm.clone()
                    self.grad_count = 0
                else:
                    self.grad_accum += grad_norm
            self.grad_count += 1
    
    def train(
        self,
        num_epochs: int,
        use_sampling: bool = False,
        num_samples: int = 500000,
        eval_interval: int = 10,
        save_interval: int = 500,
        save_dir: str = None,
        verbose: bool = True,
        use_densification: bool = False
    ):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Learning rate scheduler: 
        # - Use initial_lr during densification phase
        # - Switch to finetune_lr after densification stops, then decay
        cfg = self.config
        initial_lr = cfg.learning_rate
        finetune_lr = cfg.finetune_lr
        finetune_printed = False
        
        pbar = tqdm(range(num_epochs), desc="Sparse Training", disable=not verbose)
        
        for epoch in pbar:
            # Apply LR schedule based on phase
            if epoch >= cfg.lr_decay_start:
                # Fine-tuning phase: start from finetune_lr and decay to finetune_lr * lr_decay
                if not finetune_printed:
                    print(f"\n[Fine-tuning] Switching to lower LR: {finetune_lr} (densification stopped)")
                    finetune_printed = True
                decay_epochs = num_epochs - cfg.lr_decay_start
                progress = (epoch - cfg.lr_decay_start) / max(1, decay_epochs)
                lr = finetune_lr * (cfg.lr_decay ** progress)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            
            if use_sampling:
                total_loss, mse_loss, sparsity_loss, overlap_loss, smoothness_loss = self.compute_loss_sparse_sampled(num_samples)
            else:
                total_loss, mse_loss, sparsity_loss, overlap_loss, smoothness_loss = self.compute_loss_sparse()
            
            total_loss.backward()
            
            if use_densification:
                self.accumulate_gradients()
            
            self.optimizer.step()
            
            n_split, n_clone, n_prune = 0, 0, 0
            if use_densification:
                n_split, n_clone, n_prune = self.densify_step(epoch)
            
            self.history['total_loss'].append(total_loss.item())
            self.history['mse_loss'].append(mse_loss.item())
            self.history['sparsity_loss'].append(sparsity_loss.item())
            self.history['overlap_loss'].append(overlap_loss.item())
            self.history['smoothness_loss'].append(smoothness_loss.item())
            self.history['num_gaussians'].append(self.model.N)
            self.history['densify_split'].append(n_split)
            self.history['densify_clone'].append(n_clone)
            self.history['densify_prune'].append(n_prune)
            
            if epoch % eval_interval == 0:
                psnr = self.compute_psnr_sparse()
                self.history['psnr'].append(psnr)
                
                # SSIM/LPIPS disabled - too slow for large volumes
                # Only compute at final epoch if needed
                pbar.set_postfix({
                        'loss': f'{total_loss.item():.6f}',
                        'psnr': f'{psnr:.2f}',
                        'N': self.model.N
                    })
            
            if save_dir and save_interval > 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_dir, epoch + 1)
        
        # Save final checkpoint
        if save_dir:
            self.save_checkpoint(save_dir, num_epochs, is_final=True)
        
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int, is_final: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_params': self.model.get_parameters_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'learning_rate': self.config.learning_rate,
                'lambda_sparsity': self.config.lambda_sparsity,
                'lambda_overlap': self.config.lambda_overlap,
                'lambda_smoothness': self.config.lambda_smoothness,
                'knn_k': self.config.knn_k,
                'densify_grad_threshold': self.config.densify_grad_threshold,
                'densify_scale_threshold': self.config.densify_scale_threshold,
                'prune_intensity_threshold': self.config.prune_intensity_threshold,
                'prune_scale_threshold': self.config.prune_scale_threshold,
                'max_gaussians': self.config.max_gaussians,
            },
            'output_dir': self.output_dir,
        }
        
        if is_final:
            path = os.path.join(save_dir, 'checkpoint_final.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:06d}.pt')
        
        torch.save(checkpoint, path)
        
        # Also save a compact .ply-like format for visualization
        if is_final:
            self._save_gaussians_compact(save_dir)
    
    def _save_gaussians_compact(self, save_dir: str):
        """Save Gaussians in a compact format for visualization."""
        import json
        
        with torch.no_grad():
            data = {
                'num_gaussians': self.model.N,
                'volume_shape': list(self.model.volume_shape),
                'positions': self.model.positions.cpu().numpy().tolist(),
                'scales': self.model.scales.cpu().numpy().tolist(),
                'rotations': self.model.rotations.cpu().numpy().tolist(),
                'intensities': self.model.intensities().cpu().numpy().tolist(),
            }
        
        path = os.path.join(save_dir, 'gaussians.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved Gaussians to {path}")
    
    def reconstruct_full_volume(self):
        """Reconstruct full volume for visualization/saving."""
        with torch.no_grad():
            return self.model()


def main():
    parser = argparse.ArgumentParser(description='Sparse Gate-Guided Gaussian Training')
    parser.add_argument('--volume', type=str, required=True, help='Path to volume TIFF')
    parser.add_argument('--gate_checkpoint', type=str, default=None, help='Path to TOPS-Gate checkpoint')
    parser.add_argument('--gate_mask', type=str, default=None, help='Path to pre-computed gate mask TIFF (alternative to gate_checkpoint)')
    parser.add_argument('--gate_tau', type=float, default=0.5, help='Gate threshold')
    parser.add_argument('--num_gaussians', type=int, default=20000, help='Number of Gaussians')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='output/sparse_gate_gaussian', help='Output directory')
    parser.add_argument('--densify', action='store_true', help='Enable densification')
    parser.add_argument('--max_gaussians', type=int, default=50000, help='Max Gaussians')
    parser.add_argument('--eval_interval', type=int, default=10, help='PSNR eval interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Checkpoint save interval')
    parser.add_argument('--use_sampling', action='store_true', help='Sample from gated voxels (faster)')
    parser.add_argument('--sample_ratio', type=float, default=0.2, help='Fraction of gated voxels to sample (0.2 = 20%)')
    parser.add_argument('--num_samples', type=int, default=0, help='Fixed samples per iteration (0 = use sample_ratio instead)')
    parser.add_argument('--knn_k', type=int, default=32, help='K nearest Gaussians per query (lower=faster, higher=more accurate)')
    parser.add_argument('--lambda_sparsity', type=float, default=0.001, help='Sparsity regularization weight')
    parser.add_argument('--lambda_overlap', type=float, default=0.001, help='Overlap regularization weight (0=disabled)')
    parser.add_argument('--lambda_smoothness', type=float, default=0.001, help='Smoothness regularization weight (0=disabled)')
    parser.add_argument('--smoothness_neighbors', type=int, default=5, help='k-NN for smoothness regularization')
    parser.add_argument('--edge_boost', type=float, default=3.0, help='Edge loss weighting boost (0=disabled)')
    parser.add_argument('--no_edge_weights', action='store_true', help='Disable edge-aware loss weighting')
    parser.add_argument('--grad_threshold', type=float, default=0.0001, help='Gradient threshold for densification (lower=more densification)')
    
    args = parser.parse_args()
    device = 'cuda'
    
    # Create versioned output directory
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Find next version number
    existing_versions = [d for d in os.listdir(base_output_dir) 
                        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith('v')]
    if existing_versions:
        version_nums = []
        for v in existing_versions:
            try:
                version_nums.append(int(v[1:]))
            except ValueError:
                pass
        next_version = max(version_nums) + 1 if version_nums else 1
    else:
        next_version = 1
    
    version_dir = os.path.join(base_output_dir, f'v{next_version:03d}')
    os.makedirs(version_dir, exist_ok=True)
    args.output_dir = version_dir
    
    print("=" * 60)
    print("Sparse Gate-Guided Gaussian Training")
    print(f"Output: {version_dir}")
    print("=" * 60)
    
    # Load volume
    print(f"\nLoading volume: {args.volume}")
    volume = tifffile.imread(args.volume).astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    volume_shape = volume.shape
    total_voxels = np.prod(volume_shape)
    print(f"  Shape: {volume_shape}, total: {total_voxels:,} voxels")
    
    # Save GT
    gt_path = os.path.join(args.output_dir, 'ground_truth.tif')
    tifffile.imwrite(gt_path, volume)
    
    # Load gate mask (either from pre-computed file or generate from checkpoint)
    if args.gate_mask is not None:
        # Load pre-computed gate mask
        print(f"\nLoading pre-computed gate mask from {args.gate_mask}...")
        gate_mask_np = tifffile.imread(args.gate_mask)
        gate_mask = torch.from_numpy(gate_mask_np > 0).to(device)
        print(f"  Gate mask shape: {gate_mask.shape}")
    elif args.gate_checkpoint is not None:
        # Generate from TOPS-Gate model
        gate_model = load_tops_gate_model(args.gate_checkpoint, device)
        gate_mask = generate_hard_gate_mask(gate_model, volume_shape, tau=args.gate_tau, device=device)
        # Save gate mask
        tifffile.imwrite(os.path.join(args.output_dir, 'gate_mask.tif'), 
                         gate_mask.cpu().numpy().astype(np.uint8) * 255)
    else:
        raise ValueError("Must provide either --gate_checkpoint or --gate_mask")
    
    n_gated = int(gate_mask.sum().item())
    
    # Initialize Gaussians in gated regions
    gated_indices = torch.nonzero(gate_mask, as_tuple=False)
    if args.num_gaussians > n_gated:
        print(f"  Reducing Gaussians from {args.num_gaussians} to {n_gated} (all gated voxels)")
        args.num_gaussians = n_gated
    
    perm = torch.randperm(n_gated, device=device)[:args.num_gaussians]
    selected_indices = gated_indices[perm]
    
    D, H, W = volume_shape
    init_positions = torch.zeros(args.num_gaussians, 3, device=device)
    init_positions[:, 0] = (selected_indices[:, 0].float() + 0.5) / D
    init_positions[:, 1] = (selected_indices[:, 1].float() + 0.5) / H
    init_positions[:, 2] = (selected_indices[:, 2].float() + 0.5) / W
    
    # Create model
    print(f"\nCreating Gaussian model with {args.num_gaussians} Gaussians...")
    model = CUDAGaussianModel(
        num_gaussians=args.num_gaussians,
        volume_shape=volume_shape,
        init_method='uniform',
        device=device
    )
    
    # Set gate-guided positions
    with torch.no_grad():
        eps = 1e-6
        init_positions_clamped = init_positions.clamp(eps, 1.0 - eps)
        raw_positions = torch.log(init_positions_clamped / (1.0 - init_positions_clamped))
        model.positions_raw.data = raw_positions
    
    # Create sparse trainer
    config = SparseTrainerConfig(
        learning_rate=args.lr,
        lambda_sparsity=args.lambda_sparsity,
        lambda_overlap=args.lambda_overlap,
        lambda_smoothness=args.lambda_smoothness,
        smoothness_neighbors=args.smoothness_neighbors,
        knn_k=args.knn_k,
        max_gaussians=args.max_gaussians,
        use_edge_weights=not args.no_edge_weights,
        edge_boost=args.edge_boost,
        densify_grad_threshold=args.grad_threshold,
    )
    
    volume_tensor = torch.from_numpy(volume).float()
    
    trainer = SparseGateGuidedTrainer(
        model=model,
        volume=volume_tensor,
        gate_mask=gate_mask,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    # Compute num_samples from ratio if not specified
    if args.use_sampling:
        if args.num_samples > 0:
            num_samples = args.num_samples
        else:
            num_samples = int(n_gated * args.sample_ratio)
        print(f"\nSampling {args.sample_ratio*100:.0f}% of gated voxels = {num_samples:,} samples/epoch")
    else:
        num_samples = n_gated  # Use all
    
    # Train
    print(f"\nStarting SPARSE training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Using sampling: {args.use_sampling}")
    if args.use_sampling:
        print(f"  Samples per iter: {num_samples:,} ({args.sample_ratio*100:.0f}% of {n_gated:,} gated voxels)")
    print(f"  Loss weights: sparsity={args.lambda_sparsity}, overlap={args.lambda_overlap}, smoothness={args.lambda_smoothness}")
    
    # Save initial config (in case training crashes)
    from datetime import datetime
    initial_config = {
        'version': f'v{next_version:03d}',
        'timestamp': datetime.now().isoformat(),
        'status': 'training',
        'args': vars(args),
        'data': {
            'volume_shape': list(volume_shape),
            'total_voxels': int(total_voxels),
            'gated_voxels': int(n_gated),
            'gate_occupancy_pct': float(100 * n_gated / total_voxels),
        },
        'densification': {
            'enabled': args.densify,
            'grad_threshold': config.densify_grad_threshold,
            'scale_threshold': config.densify_scale_threshold,
            'interval': config.densify_interval,
            'start': config.densify_start,
            'stop': config.densify_stop,
        },
        'pruning': {
            'intensity_threshold': config.prune_intensity_threshold,
            'scale_threshold': config.prune_scale_threshold,
        },
        'sample_ratio': args.sample_ratio,
        'samples_per_iter': num_samples,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(initial_config, f, indent=2)
    print(f"  Config saved to {args.output_dir}/config.json")
    
    history = trainer.train(
        num_epochs=args.epochs,
        use_sampling=args.use_sampling,
        num_samples=num_samples,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
        verbose=True,
        use_densification=args.densify
    )
    
    # Save results
    print("\nSaving results...")
    
    # Reconstructed volume (full, for visualization)
    with torch.no_grad():
        recon = trainer.reconstruct_full_volume()
        if trainer.volume_max > 0:
            recon = recon * trainer.volume_max
        tifffile.imwrite(os.path.join(args.output_dir, 'reconstructed.tif'), 
                         recon.cpu().numpy().astype(np.float32))
    
    # History
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training configuration and parameters
    from datetime import datetime
    config_log = {
        'version': f'v{next_version:03d}',
        'timestamp': datetime.now().isoformat(),
        'args': {
            'volume': args.volume,
            'gate_checkpoint': args.gate_checkpoint,
            'gate_mask': args.gate_mask,
            'gate_tau': args.gate_tau,
            'num_gaussians': args.num_gaussians,
            'epochs': args.epochs,
            'lr': args.lr,
            'densify': args.densify,
            'max_gaussians': args.max_gaussians,
            'eval_interval': args.eval_interval,
            'save_interval': args.save_interval,
            'use_sampling': args.use_sampling,
            'num_samples': args.num_samples,
            'knn_k': args.knn_k,
            'lambda_sparsity': args.lambda_sparsity,
            'lambda_overlap': args.lambda_overlap,
            'lambda_smoothness': args.lambda_smoothness,
            'smoothness_neighbors': args.smoothness_neighbors,
            'edge_boost': args.edge_boost,
            'use_edge_weights': not args.no_edge_weights,
            'grad_threshold': args.grad_threshold,
        },
        'data': {
            'volume_shape': list(volume_shape),
            'total_voxels': int(total_voxels),
            'gated_voxels': int(n_gated),
            'gate_occupancy_pct': float(100 * n_gated / total_voxels),
        },
        'results': {
            'final_psnr': history['psnr'][-1] if history['psnr'] else None,
            'final_ssim': history['ssim'][-1] if history['ssim'] else None,
            'final_lpips': history['lpips'][-1] if history['lpips'] else None,
            'final_loss': history['total_loss'][-1] if history['total_loss'] else None,
            'final_gaussians': model.N,
            'best_psnr': max(history['psnr']) if history['psnr'] else None,
            'best_ssim': max(history['ssim']) if history['ssim'] else None,
            'best_lpips': min(history['lpips']) if history['lpips'] else None,
        }
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config_log, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Sparse Gate-Guided Training Complete!")
    print("=" * 60)
    if history['psnr']:
        print(f"  Final PSNR (on gated voxels): {history['psnr'][-1]:.2f} dB")
    if history['ssim']:
        print(f"  Final SSIM (MIP avg): {history['ssim'][-1]:.4f}")
    if history['lpips']:
        print(f"  Final LPIPS (MIP avg): {history['lpips'][-1]:.4f}")
    print(f"  Final Loss: {history['total_loss'][-1]:.6f}")
    print(f"  Final Gaussians: {model.N}")
    print(f"  Voxels used: {n_gated:,} / {total_voxels:,} ({100*n_gated/total_voxels:.2f}%)")
    print(f"  Results: {args.output_dir}")


if __name__ == '__main__':
    main()
