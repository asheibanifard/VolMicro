#!/usr/bin/env python3
"""
Load and use trained CUDA Gaussian Volume Model

Usage:
    python load_model.py --checkpoint <path_to_checkpoint.pth> --volume <original_volume.tif>
"""

import argparse
import os
import sys
import torch
import numpy as np
import tifffile as tiff

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_model_cuda import CUDAGaussianModel


def load_model(checkpoint_path: str, volume_shape: tuple = None, device: str = 'cuda') -> CUDAGaussianModel:
    """
    Load trained Gaussian model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        volume_shape: Volume shape (D, H, W). If None, loaded from checkpoint.
        device: Device to use
        
    Returns:
        Loaded CUDAGaussianModel
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters from checkpoint
    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        num_gaussians = params['num_gaussians']
        vol_shape = tuple(params['volume_shape'])
    else:
        # Try to infer from state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        num_gaussians = state_dict['positions'].shape[0]
        if volume_shape is None:
            raise ValueError("volume_shape must be provided if not in checkpoint")
        vol_shape = volume_shape
    
    print(f"  Number of Gaussians: {num_gaussians}")
    print(f"  Volume shape: {vol_shape}")
    
    # Create model
    model = CUDAGaussianModel(
        num_gaussians=num_gaussians,
        volume_shape=vol_shape,
        device=device
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_params' in checkpoint:
        # New checkpoint format: model_params contains both metadata and tensors
        params = checkpoint['model_params']
        state_dict = {
            'positions_raw': params['positions_raw'],
            'scales_raw': params['scales_raw'],
            'rotations': params['rotations'],
            'intensities_raw': params['intensities_raw'],
        }
        # Handle grid_points if it exists in model but not in checkpoint
        if hasattr(model, 'grid_points') and 'grid_points' not in state_dict:
            state_dict['grid_points'] = model.grid_points
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("  Model loaded successfully!")
    
    # Print training info if available
    if 'epoch' in checkpoint:
        print(f"  Trained for: {checkpoint['epoch']} epochs")
    if 'history' in checkpoint and checkpoint['history'].get('psnr'):
        print(f"  Final PSNR: {checkpoint['history']['psnr'][-1]:.2f} dB")
    
    return model


def reconstruct_volume(model: CUDAGaussianModel, volume_max: float = 1.0) -> torch.Tensor:
    """
    Reconstruct volume from Gaussian model.
    
    Args:
        model: Trained CUDAGaussianModel
        volume_max: Maximum value for unnormalization
        
    Returns:
        Reconstructed volume tensor
    """
    print("Reconstructing volume...")
    with torch.no_grad():
        recon = model.reconstruct_volume()
        if volume_max > 1.0:
            recon = recon * volume_max
    print(f"  Reconstructed shape: {recon.shape}")
    print(f"  Value range: [{recon.min():.2f}, {recon.max():.2f}]")
    return recon


def compute_metrics(recon: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute reconstruction metrics."""
    mse = torch.mean((recon - target) ** 2).item()
    psnr = 20 * np.log10(target.max().item() / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # SSIM-like metric (simplified)
    recon_norm = (recon - recon.mean()) / (recon.std() + 1e-8)
    target_norm = (target - target.mean()) / (target.std() + 1e-8)
    correlation = torch.mean(recon_norm * target_norm).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': correlation
    }


def main():
    parser = argparse.ArgumentParser(description='Load trained Gaussian Volume Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--volume', type=str, default=None,
                        help='Path to original volume (for comparison)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for reconstructed volume')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load original volume if provided
    volume_shape = None
    volume = None
    volume_max = 1.0
    
    if args.volume:
        print(f"Loading original volume: {args.volume}")
        volume = tiff.imread(args.volume)
        volume = torch.tensor(volume, dtype=torch.float32, device=args.device)
        volume_shape = tuple(volume.shape)
        volume_max = volume.max().item()
        print(f"  Shape: {volume_shape}")
        print(f"  Range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Load model
    model = load_model(args.checkpoint, volume_shape, args.device)
    
    # Reconstruct
    recon = reconstruct_volume(model, volume_max)
    
    # Compare if original volume provided
    if volume is not None:
        metrics = compute_metrics(recon, volume)
        print(f"\nReconstruction Metrics:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Correlation: {metrics['correlation']:.4f}")
    
    # Save reconstruction
    if args.output:
        output_path = args.output
    else:
        # Default output next to checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(checkpoint_dir, 'loaded_reconstruction.tif')
    
    print(f"\nSaving reconstruction to: {output_path}")
    recon_np = recon.cpu().numpy()
    
    # Clip to valid range and convert to uint8 for compatibility
    recon_np = np.clip(recon_np, 0, 255).astype(np.uint8)
    print(f"  Output range: [{recon_np.min()}, {recon_np.max()}]")
    print(f"  Output dtype: {recon_np.dtype}")
    
    tiff.imwrite(output_path, recon_np)
    
    print("\nDone!")
    
    return model, recon


if __name__ == '__main__':
    main()
