#!/usr/bin/env python3
"""
CUDA-Accelerated Training Script for Gaussian-Based Volume Data Representation

Uses custom CUDA kernels for fast Gaussian evaluation.
Based on [21 Dec. 24] Algorithm.

Usage:
    python train_cuda.py --volume <volume.tif> --num_gaussians 5000 --epochs 100
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tifffile as tiff
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from tqdm import tqdm

from gaussian_model_cuda import CUDAGaussianModel


@dataclass
class TrainerConfig:
    """Configuration for CUDA Gaussian trainer."""
    learning_rate: float = 0.01
    lambda_sparsity: float = 0.001
    densify_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.05
    densify_interval: int = 100
    densify_start: int = 500
    densify_stop: int = 15000
    max_gaussians: int = 50000


class CUDAGaussianTrainer:
    """
    CUDA-accelerated trainer for Gaussian volume model.
    """
    
    def __init__(
        self,
        model: CUDAGaussianModel,
        volume: torch.Tensor,
        config: TrainerConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.config = config
        
        # Normalize volume
        self.volume = volume.to(device)
        self.volume_max = self.volume.max().item()
        if self.volume_max > 0:
            self.volume_norm = self.volume / self.volume_max
        else:
            self.volume_norm = self.volume
        
        self.volume_flat = self.volume_norm.flatten()
        self.D, self.H, self.W = self.volume.shape
        
        # Create coordinate grid for sampling
        d = (torch.arange(self.D, device=device, dtype=torch.float32) + 0.5) / self.D
        h = (torch.arange(self.H, device=device, dtype=torch.float32) + 0.5) / self.H
        w = (torch.arange(self.W, device=device, dtype=torch.float32) + 0.5) / self.W
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        self.all_coords = torch.stack([
            grid_d.flatten(), grid_h.flatten(), grid_w.flatten()
        ], dim=1)  # (D*H*W, 3)
        
        self.total_voxels = self.D * self.H * self.W
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # History
        self.history = {
            'total_loss': [],
            'mse_loss': [],
            'sparsity_loss': [],
            'psnr': [],
            'num_gaussians': [],
            'densify_split': [],
            'densify_clone': [],
            'densify_prune': [],
        }
        
        # Gradient accumulator for densification
        self.grad_accum = None
        self.grad_count = 0
    
    def sample_batch(self, batch_size: int):
        """Sample random voxels."""
        indices = torch.randint(0, self.total_voxels, (batch_size,), device=self.device)
        coords = self.all_coords[indices]
        values = self.volume_flat[indices]
        return coords, values, indices
    
    def compute_loss_sampled(self, num_samples: int):
        """Compute loss using random sampling."""
        coords, gt_values, _ = self.sample_batch(num_samples)
        pred_values = self.model.forward_sampled(coords)
        
        mse_loss = nn.functional.mse_loss(pred_values, gt_values)
        
        # Sparsity regularization
        intensities = self.model.intensities()
        sparsity_loss = self.config.lambda_sparsity * intensities.abs().mean()
        
        total_loss = mse_loss + sparsity_loss
        return total_loss, mse_loss, sparsity_loss
    
    def compute_loss_full(self):
        """Compute loss on full volume."""
        pred_volume = self.model()  # (D, H, W)
        mse_loss = nn.functional.mse_loss(pred_volume, self.volume_norm)
        
        intensities = self.model.intensities()
        sparsity_loss = self.config.lambda_sparsity * intensities.abs().mean()
        
        total_loss = mse_loss + sparsity_loss
        return total_loss, mse_loss, sparsity_loss
    
    def compute_psnr(self):
        """Compute PSNR on full volume."""
        with torch.no_grad():
            pred = self.model()
            mse = nn.functional.mse_loss(pred, self.volume_norm)
            if mse < 1e-10:
                return float('inf')
            return (10 * torch.log10(1.0 / mse)).item()
    
    def densify_step(self, iteration: int):
        """Perform densification (split/clone/prune) based on gradients."""
        cfg = self.config
        
        if iteration < cfg.densify_start or iteration > cfg.densify_stop:
            return 0, 0, 0
        if iteration % cfg.densify_interval != 0:
            return 0, 0, 0
        if self.grad_accum is None or self.grad_count == 0:
            return 0, 0, 0
        
        # Average gradients
        avg_grads = self.grad_accum / self.grad_count
        
        # Clone small Gaussians with high gradients
        n_clone = 0
        if self.model.N < cfg.max_gaussians:
            n_clone = self.model.densify_and_clone(
                avg_grads, cfg.densify_grad_threshold, cfg.densify_scale_threshold
            )
        
        # Split large Gaussians with high gradients  
        n_split = 0
        if self.model.N < cfg.max_gaussians:
            n_split = self.model.densify_and_split(
                avg_grads, cfg.densify_grad_threshold, cfg.densify_scale_threshold
            )
        
        # Prune low-contribution Gaussians (DISABLED - causes instability)
        # n_prune = self.model.prune_gaussians(intensity_threshold=0.005)
        n_prune = 0
        
        # Reset optimizer and gradient accumulator
        if n_clone > 0 or n_split > 0 or n_prune > 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
            self.grad_accum = None
            self.grad_count = 0
        
        return n_split, n_clone, n_prune
    
    def accumulate_gradients(self):
        """Accumulate position gradients for densification."""
        if self.model.positions_raw.grad is not None:
            grad_norm = self.model.positions_raw.grad.norm(dim=1)
            if self.grad_accum is None:
                self.grad_accum = grad_norm.clone()
            else:
                # Handle size mismatch after densification
                if self.grad_accum.shape[0] != grad_norm.shape[0]:
                    self.grad_accum = grad_norm.clone()
                    self.grad_count = 0
                else:
                    self.grad_accum += grad_norm
            self.grad_count += 1
    
    def train(
        self,
        num_epochs: int,
        use_sampling: bool = True,
        num_samples: int = 100000,
        eval_interval: int = 1,
        save_interval: int = 50,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        use_densification: bool = False
    ) -> Dict:
        """
        Train the Gaussian model.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            if use_sampling:
                total_loss, mse_loss, sparsity_loss = self.compute_loss_sampled(num_samples)
            else:
                total_loss, mse_loss, sparsity_loss = self.compute_loss_full()
            
            total_loss.backward()
            
            # Accumulate gradients for densification
            if use_densification:
                self.accumulate_gradients()
            
            self.optimizer.step()
            
            # Densification
            n_split, n_clone, n_prune = 0, 0, 0
            if use_densification:
                n_split, n_clone, n_prune = self.densify_step(epoch)
            
            # Record history
            self.history['total_loss'].append(total_loss.item())
            self.history['mse_loss'].append(mse_loss.item())
            self.history['sparsity_loss'].append(sparsity_loss.item())
            self.history['num_gaussians'].append(self.model.N)
            self.history['densify_split'].append(n_split)
            self.history['densify_clone'].append(n_clone)
            self.history['densify_prune'].append(n_prune)
            
            # Evaluate PSNR
            if epoch % eval_interval == 0:
                psnr = self.compute_psnr()
                self.history['psnr'].append(psnr)
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.6f}',
                    'psnr': f'{psnr:.2f}',
                    'N': self.model.N
                })
            
            # Save checkpoint
            if save_dir and save_interval > 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_dir, epoch + 1)
        
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_params': self.model.get_parameters_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
        }
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:06d}.pt')
        torch.save(checkpoint, path)


def load_volume(path: str) -> torch.Tensor:
    """Load volume from TIFF file."""
    print(f"Loading volume from: {path}")
    volume = tiff.imread(path)
    volume = torch.tensor(volume, dtype=torch.float32)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    return volume


def save_results(
    model: CUDAGaussianModel,
    trainer: CUDAGaussianTrainer,
    output_dir: str,
    volume: torch.Tensor
):
    """Save training results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reconstructed volume
    with torch.no_grad():
        recon = model()  # forward() returns (D, H, W) volume
        # Unnormalize
        if trainer.volume_max > 0:
            recon = recon * trainer.volume_max
        recon_np = recon.cpu().numpy().astype(np.float32)
        tiff.imwrite(os.path.join(output_dir, 'reconstructed.tif'), recon_np)
    
    # Save training history
    import json
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    # Save model parameters
    params = model.get_parameters_dict()
    params_save = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                   for k, v in params.items()}
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(params_save, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='CUDA-Accelerated Gaussian Volume Training'
    )
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to volume TIFF file')
    parser.add_argument('--num_gaussians', type=int, default=5000,
                        help='Number of Gaussian basis functions')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lambda_sparsity', type=float, default=0.001,
                        help='Sparsity regularization weight')
    parser.add_argument('--init_method', type=str, default='uniform',
                        choices=['uniform', 'grid', 'swc'],
                        help='Initialization method')
    parser.add_argument('--swc_path', type=str, default=None,
                        help='Path to SWC file for skeleton initialization')
    parser.add_argument('--no_swc_densify', action='store_true',
                        help='Do not densify SWC skeleton (use raw SWC nodes only)')
    parser.add_argument('--use_sampling', action='store_true',
                        help='Use random sampling for loss (faster)')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples when using sampling')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluate every N epochs')
    
    # Densification arguments
    parser.add_argument('--densify', action='store_true',
                        help='Enable adaptive density control (split/clone/prune)')
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002,
                        help='Gradient threshold for densification')
    parser.add_argument('--densify_scale_threshold', type=float, default=0.05,
                        help='Scale threshold for split vs clone')
    parser.add_argument('--densify_interval', type=int, default=100,
                        help='Densify every N iterations')
    parser.add_argument('--densify_start', type=int, default=500,
                        help='Start densification after N iterations')
    parser.add_argument('--densify_stop', type=int, default=15000,
                        help='Stop densification after N iterations')
    parser.add_argument('--max_gaussians', type=int, default=50000,
                        help='Maximum number of Gaussians')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this script!")
        sys.exit(1)
    
    print("=" * 60)
    print("CUDA-Accelerated Gaussian-Based Volume Data Representation")
    print("Implementation based on [21 Dec. 24] Algorithm")
    print("=" * 60)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Load volume
    volume = load_volume(args.volume)
    volume_shape = tuple(volume.shape)
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/cuda_gaussian_{timestamp}"
    
    print(f"\nConfiguration:")
    print(f"  Number of Gaussians: {args.num_gaussians}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Sparsity weight: {args.lambda_sparsity}")
    print(f"  Use sampling: {args.use_sampling}")
    if args.use_sampling:
        print(f"  Num samples: {args.num_samples}")
    print(f"  Densification: {args.densify}")
    if args.densify:
        print(f"    Grad threshold: {args.densify_grad_threshold}")
        print(f"    Scale threshold: {args.densify_scale_threshold}")
        print(f"    Interval: {args.densify_interval}")
        print(f"    Start/Stop: {args.densify_start}/{args.densify_stop}")
        print(f"    Max Gaussians: {args.max_gaussians}")
    print(f"  Output: {args.output_dir}")
    
    # Create model (or load from checkpoint)
    print("\nInitializing model...")
    start_epoch = 0
    
    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
        params = checkpoint['model_params']
        model = CUDAGaussianModel(
            num_gaussians=params['num_gaussians'],
            volume_shape=volume_shape,
            device='cuda'
        )
        # Load raw parameters
        model.positions_raw.data = params['positions_raw'].cuda()
        model.scales_raw.data = params['scales_raw'].cuda()
        model.rotations.data = params['rotations'].cuda()
        model.intensities_raw.data = params['intensities_raw'].cuda()
        model.raw_opacities.data = params['raw_opacities'].cuda()
        model.N = params['num_gaussians']
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  Resumed from epoch {start_epoch}")
    else:
        model = CUDAGaussianModel(
            num_gaussians=args.num_gaussians,
            volume_shape=volume_shape,
            init_method=args.init_method,
            device='cuda',
            swc_path=args.swc_path,
            swc_densify=not args.no_swc_densify,
        )
    print(f"  Gaussians: {model.N}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer config (TrainerConfig is defined at module level)
    config = TrainerConfig(
        learning_rate=args.lr,
        lambda_sparsity=args.lambda_sparsity,
        densify_grad_threshold=args.densify_grad_threshold,
        densify_scale_threshold=args.densify_scale_threshold,
        densify_interval=args.densify_interval,
        densify_start=args.densify_start,
        densify_stop=args.densify_stop,
        max_gaussians=args.max_gaussians,
    )
    
    # Create trainer
    trainer = CUDAGaussianTrainer(
        model=model,
        volume=volume,
        config=config,
        device='cuda'
    )
    
    # Train
    print("\nStarting CUDA-accelerated training...")
    history = trainer.train(
        num_epochs=args.epochs,
        use_sampling=args.use_sampling,
        num_samples=args.num_samples,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
        verbose=True,
        use_densification=args.densify
    )
    
    # Save results
    save_results(model, trainer, args.output_dir, volume)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    if history['psnr']:
        print(f"  Final PSNR: {history['psnr'][-1]:.2f} dB")
    print(f"  Final Loss: {history['total_loss'][-1]:.6f}")
    print(f"  Final Gaussians: {model.N}")
    if args.densify and 'num_gaussians' in history:
        print(f"  Gaussians: {args.num_gaussians} → {model.N}")
        total_split = sum(history.get('densify_split', []))
        total_clone = sum(history.get('densify_clone', []))
        total_prune = sum(history.get('densify_prune', []))
        print(f"  Densification: Split={total_split}, Clone={total_clone}, Prune={total_prune}")


if __name__ == '__main__':
    main()
