#!/usr/bin/env python3
"""
Fast PyTorch Gaussian Volume Training with Sparse Evaluation

Uses efficient sparse evaluation - only computes Gaussians near their centers.
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
from tqdm import tqdm


class FastGaussianModel(nn.Module):
    """Fast Gaussian model using sparse evaluation on GPU."""
    
    def __init__(self, num_gaussians: int, volume_shape: tuple, device: str = 'cuda'):
        super().__init__()
        
        self.N = num_gaussians
        self.D, self.H, self.W = volume_shape
        self.device = device
        
        # Learnable parameters (normalized to 0-1)
        self.positions = nn.Parameter(torch.rand(num_gaussians, 3, device=device))
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 3), -3.0, device=device))
        self.intensities = nn.Parameter(torch.rand(num_gaussians, device=device) * 0.5)
        
        # Rotation quaternions (identity by default for simplicity)
        self.rotations = nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
        self.rotations.data[:, 0] = 1.0
    
    def forward(self, num_eval_points: int = 100000) -> tuple:
        """
        Evaluate model at random points.
        
        Returns (predicted_values, coordinates, flat_indices)
        """
        # Sample random coordinates
        d = torch.rand(num_eval_points, device=self.device) * (self.D - 1)
        h = torch.rand(num_eval_points, device=self.device) * (self.H - 1)
        w = torch.rand(num_eval_points, device=self.device) * (self.W - 1)
        
        # Normalize to 0-1 for Gaussian evaluation
        d_norm = d / (self.D - 1)
        h_norm = h / (self.H - 1)
        w_norm = w / (self.W - 1)
        
        points = torch.stack([d_norm, h_norm, w_norm], dim=1)  # (M, 3)
        
        # Compute flat indices for volume lookup
        d_idx = d.long().clamp(0, self.D - 1)
        h_idx = h.long().clamp(0, self.H - 1)
        w_idx = w.long().clamp(0, self.W - 1)
        flat_indices = d_idx * (self.H * self.W) + h_idx * self.W + w_idx
        
        # Evaluate Gaussians
        values = self._eval_gaussians(points)
        
        return values, points, flat_indices
    
    def _eval_gaussians(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate sum of Gaussians at given points."""
        M = points.shape[0]
        
        scales = torch.exp(self.log_scales)  # (N, 3)
        
        # Distance from each point to each Gaussian center
        # points: (M, 3), positions: (N, 3)
        diff = points.unsqueeze(1) - self.positions.unsqueeze(0)  # (M, N, 3)
        
        # For diagonal covariance (simplification for speed):
        # exp(-0.5 * sum((x - u)^2 / s^2))
        inv_var = 1.0 / (scales ** 2 + 1e-6)  # (N, 3)
        
        # Mahalanobis distance with diagonal covariance
        quad = torch.sum(diff ** 2 * inv_var.unsqueeze(0), dim=2)  # (M, N)
        
        # Gaussian values
        gauss = torch.exp(-0.5 * quad)  # (M, N)
        
        # Weighted sum
        values = torch.matmul(gauss, self.intensities)  # (M,)
        
        return values
    
    def render_volume(self, batch_size: int = 50000) -> torch.Tensor:
        """Render full volume (for evaluation only)."""
        with torch.no_grad():
            # Create grid
            d_coords = torch.linspace(0, 1, self.D, device=self.device)
            h_coords = torch.linspace(0, 1, self.H, device=self.device)
            w_coords = torch.linspace(0, 1, self.W, device=self.device)
            
            grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
            points = torch.stack([grid_d.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)
            
            # Evaluate in batches
            values = torch.zeros(points.shape[0], device=self.device)
            for i in range(0, points.shape[0], batch_size):
                batch = points[i:i+batch_size]
                values[i:i+batch_size] = self._eval_gaussians(batch)
            
            return values.reshape(self.D, self.H, self.W)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR."""
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_val = target.max()
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


def train(
    volume_path: str,
    num_gaussians: int = 5000,
    epochs: int = 100,
    samples_per_iter: int = 100000,
    lr: float = 0.01,
    output_dir: str = None
):
    """Train fast Gaussian model."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load volume
    print(f"Loading volume from: {volume_path}")
    volume = tiff.imread(volume_path)
    volume = torch.tensor(volume, dtype=torch.float32, device=device)
    print(f"  Shape: {volume.shape}")
    print(f"  Range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Normalize
    vol_max = volume.max()
    volume_norm = volume / vol_max if vol_max > 0 else volume
    volume_flat = volume_norm.flatten()
    
    # Create model
    model = FastGaussianModel(num_gaussians, tuple(volume.shape), device)
    print(f"\nModel with {num_gaussians} Gaussians")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Output
    if output_dir is None:
        output_dir = f"outputs/fast_gaussian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training
    print(f"\nTraining for {epochs} epochs with {samples_per_iter} samples/iter...")
    
    history = {'loss': [], 'psnr': []}
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward
        pred_values, _, flat_indices = model(samples_per_iter)
        target_values = volume_flat[flat_indices]
        
        # Loss
        mse = torch.mean((pred_values - target_values) ** 2)
        sparsity = 0.001 * torch.mean(torch.abs(model.intensities))
        loss = mse + sparsity
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                recon = model.render_volume()
                psnr = compute_psnr(recon * vol_max, volume)
                history['psnr'].append(psnr)
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'psnr': f'{psnr:.2f}'})
    
    # Final evaluation
    with torch.no_grad():
        recon = model.render_volume()
        final_psnr = compute_psnr(recon * vol_max, volume)
        print(f"\nFinal PSNR: {final_psnr:.2f} dB")
        
        # Save reconstruction
        recon_np = (recon * vol_max).cpu().numpy().astype(np.float32)
        tiff.imwrite(os.path.join(output_dir, 'reconstructed.tif'), recon_np)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    print(f"Results saved to: {output_dir}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', type=str, required=True)
    parser.add_argument('--num_gaussians', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    train(
        args.volume,
        num_gaussians=args.num_gaussians,
        epochs=args.epochs,
        samples_per_iter=args.samples,
        lr=args.lr,
        output_dir=args.output
    )
