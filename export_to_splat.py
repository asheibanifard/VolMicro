#!/usr/bin/env python3
"""
Export TOPS-GATE Gaussian model to .splat format for gsplat.js viewer.

The .splat format is a binary format with the following structure per Gaussian:
- position: 3 x float32 (x, y, z)
- scales: 3 x float32 (sx, sy, sz) - log scale
- color: 4 x uint8 (r, g, b, a) - SH DC coefficients + opacity
- rotation: 4 x uint8 (qw, qx, qy, qz) - normalized quaternion

Usage:
    python export_to_splat.py --checkpoint <checkpoint.pt> --output <output.splat>
"""

import argparse
import os
import sys
import struct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaussian_model_cuda import CUDAGaussianModel


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load checkpoint and extract Gaussian parameters."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters
    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        num_gaussians = params['num_gaussians']
        vol_shape = tuple(params['volume_shape'])
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        num_gaussians = state_dict['positions_raw'].shape[0]
        vol_shape = (64, 64, 64)  # Default if not stored
    
    print(f"  Number of Gaussians: {num_gaussians}")
    print(f"  Volume shape: {vol_shape}")
    
    # Create model and load weights
    model = CUDAGaussianModel(
        num_gaussians=num_gaussians,
        volume_shape=vol_shape,
        device=device
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, vol_shape


def export_to_splat(model: CUDAGaussianModel, output_path: str, 
                    scale_factor: float = 1.0, center_offset: tuple = None):
    """
    Export Gaussian model to .splat binary format.
    
    Args:
        model: Trained CUDAGaussianModel
        output_path: Path to output .splat file
        scale_factor: Scale positions from [0,1] to world coordinates
        center_offset: (x, y, z) offset to center the model
    """
    with torch.no_grad():
        # Get Gaussian parameters
        positions = model.positions.cpu().numpy()  # (N, 3) in [0, 1]
        scales = model.scales.cpu().numpy()  # (N, 3)
        rotations = model.rotations.cpu().numpy()  # (N, 4) quaternions
        intensities = model.intensities().cpu().numpy()  # (N,)
        opacities = model.opacities.cpu().numpy()  # (N,)
    
    N = positions.shape[0]
    print(f"Exporting {N} Gaussians to: {output_path}")
    
    # Transform positions from [0,1] to centered world coordinates
    # gsplat.js expects positions in world space
    if center_offset is None:
        center_offset = (-0.5 * scale_factor, -0.5 * scale_factor, -0.5 * scale_factor)
    
    positions = positions * scale_factor + np.array(center_offset)
    
    # Scale the Gaussian sizes accordingly
    scales = scales * scale_factor
    
    # Convert to log scale (gsplat.js expects log scale)
    log_scales = np.log(scales + 1e-8)
    
    # Normalize quaternions
    quat_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (quat_norm + 1e-8)
    
    # Convert intensities to RGB color
    # For grayscale microscopy, map intensity to white/gray
    intensities_clipped = np.clip(intensities, 0, 1)
    
    # Create RGB from intensity (grayscale)
    colors_r = (intensities_clipped * 255).astype(np.uint8)
    colors_g = (intensities_clipped * 255).astype(np.uint8)
    colors_b = (intensities_clipped * 255).astype(np.uint8)
    colors_a = (np.clip(opacities, 0, 1) * 255).astype(np.uint8)
    
    # Convert quaternions to uint8 (normalized to [0, 255])
    # Quaternions are in range [-1, 1], map to [0, 255]
    quat_uint8 = ((rotations + 1.0) * 0.5 * 255).astype(np.uint8)
    
    # Write binary .splat file
    # Format: position (3xf32), log_scale (3xf32), color (4xu8), rotation (4xu8)
    # Total: 12 + 12 + 4 + 4 = 32 bytes per Gaussian
    
    with open(output_path, 'wb') as f:
        for i in range(N):
            # Position (3 x float32)
            f.write(struct.pack('fff', positions[i, 0], positions[i, 1], positions[i, 2]))
            # Log scales (3 x float32)
            f.write(struct.pack('fff', log_scales[i, 0], log_scales[i, 1], log_scales[i, 2]))
            # Color RGBA (4 x uint8)
            f.write(struct.pack('BBBB', colors_r[i], colors_g[i], colors_b[i], colors_a[i]))
            # Rotation quaternion (4 x uint8) - order: w, x, y, z
            f.write(struct.pack('BBBB', quat_uint8[i, 0], quat_uint8[i, 1], 
                               quat_uint8[i, 2], quat_uint8[i, 3]))
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.1f} KB ({file_size / N:.0f} bytes/Gaussian)")
    print(f"  Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"  Intensity range: [{intensities.min():.3f}, {intensities.max():.3f}]")
    print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export TOPS-GATE model to .splat format')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to checkpoint file (.pt or .pth)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .splat file path (default: same name as checkpoint)')
    parser.add_argument('--scale', '-s', type=float, default=2.0,
                        help='Scale factor for positions (default: 2.0)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        base = os.path.splitext(args.checkpoint)[0]
        args.output = base + '.splat'
    
    # Load model
    model, vol_shape = load_checkpoint(args.checkpoint, args.device)
    
    # Export to .splat
    export_to_splat(model, args.output, scale_factor=args.scale)
    
    print(f"\nDone! You can now view the model at:")
    print(f"  1. Copy {args.output} to gsplat.js/examples/vanilla-js/")
    print(f"  2. Add it to the volume selector dropdown")
    print(f"  3. Or load it via the file-loader example")


if __name__ == '__main__':
    main()
