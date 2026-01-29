#!/usr/bin/env python3
"""
Standalone exporter - converts TOPS-GATE checkpoint to .splat format.
Does NOT import the model class, just reads the checkpoint directly.

Usage:
    python export_to_splat_standalone.py --checkpoint <checkpoint.pt> --output <output.splat>
"""

import argparse
import os
import struct
import numpy as np
import torch
import torch.nn.functional as F


def load_gaussian_params(checkpoint_path: str, device: str = 'cpu'):
    """Load Gaussian parameters directly from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Find the parameters - could be in model_state_dict, model_params, or root
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_params' in checkpoint and 'positions_raw' in checkpoint['model_params']:
        state_dict = checkpoint['model_params']
    else:
        state_dict = checkpoint
    
    # Extract raw parameters (handle both tensor and non-tensor storage)
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)
    
    positions_raw = to_tensor(state_dict['positions_raw'])  # (N, 3)
    scales_raw = to_tensor(state_dict['scales_raw'])  # (N, 3)
    rotations = to_tensor(state_dict['rotations'])  # (N, 4)
    intensities_raw = to_tensor(state_dict['intensities_raw'])  # (N,)
    raw_opacities = to_tensor(state_dict['raw_opacities'])  # (N,)
    
    N = positions_raw.shape[0]
    print(f"  Number of Gaussians: {N}")
    
    # Get volume shape
    volume_shape = None
    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        volume_shape = params.get('volume_shape', None)
        print(f"  Volume shape: {volume_shape}")
        print(f"  Intensity activation: {params.get('intensity_activation', 'sigmoid')}")
    
    # Apply activations to get final values
    positions = torch.sigmoid(positions_raw).numpy()  # (N, 3) in [0, 1]
    scales = (F.softplus(scales_raw) + 1e-6).numpy()  # (N, 3) > 0
    rotations = rotations.numpy()  # (N, 4)
    
    # Default to sigmoid for intensities
    intensities = torch.sigmoid(intensities_raw).numpy()  # (N,) in [0, 1]
    opacities = torch.sigmoid(raw_opacities).numpy()  # (N,) in [0, 1]
    
    return positions, scales, rotations, intensities, opacities, volume_shape


def export_to_splat(positions, scales, rotations, intensities, opacities,
                    output_path: str, scale_factor: float = 2.0, volume_shape: tuple = None):
    """
    Export to .splat binary format for gsplat.js.
    
    Format per Gaussian (32 bytes total):
    - position: 3 x float32 (12 bytes) - world coordinates
    - scales: 3 x float32 (12 bytes) - DIRECT scales (not log!)
    - color: 4 x uint8 (4 bytes) - RGBA
    - rotation: 4 x uint8 (4 bytes) - quaternion wxyz, encoded as (q * 128 + 128)
    """
    N = positions.shape[0]
    print(f"Exporting {N} Gaussians to: {output_path}")
    
    # Use original volume dimensions if available
    if volume_shape is not None:
        D, H, W = volume_shape  # (100, 647, 813)
        # Scale positions from [0,1] to actual voxel coordinates, then normalize
        # to keep proportions but fit in a reasonable viewing size
        max_dim = max(D, H, W)
        dim_scale = np.array([D, H, W]) / max_dim * scale_factor
        
        # positions[:, 0] is Z (depth), [:, 1] is Y (height), [:, 2] is X (width)
        positions_world = (positions - 0.5) * dim_scale
        scales_world = scales * dim_scale
        print(f"  Using original dimensions: D={D}, H={H}, W={W}")
        print(f"  Dimension scale factors: {dim_scale}")
    else:
        # Fallback: simple centered scaling
        center = np.array([0.5, 0.5, 0.5])
        positions_world = (positions - center) * scale_factor
        scales_world = scales * scale_factor
    
    # Normalize quaternions
    quat_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations_norm = rotations / (quat_norm + 1e-8)
    
    # Convert intensities to grayscale RGB - boost for visibility
    # Normalize intensities to use full range
    intensities_min = intensities.min()
    intensities_max = intensities.max()
    if intensities_max > intensities_min:
        intensities_norm = (intensities - intensities_min) / (intensities_max - intensities_min)
    else:
        intensities_norm = intensities
    
    # Apply gamma correction to brighten mid-tones
    intensities_bright = np.power(intensities_norm, 0.5)  # gamma 0.5 brightens
    
    colors_r = (intensities_bright * 255).astype(np.uint8)
    colors_g = (intensities_bright * 255).astype(np.uint8)
    colors_b = (intensities_bright * 255).astype(np.uint8)
    
    # Boost opacity significantly for visibility (original was 0.1)
    opacities_boosted = np.clip(opacities * 8.0, 0.3, 1.0)  # Boost and ensure minimum
    colors_a = (opacities_boosted * 255).astype(np.uint8)
    
    # Convert quaternions to uint8: gsplat.js uses (q * 128 + 128)
    # Quaternion order in file: w, x, y, z
    quat_uint8 = ((rotations_norm * 128) + 128).clip(0, 255).astype(np.uint8)
    
    # Write binary file
    with open(output_path, 'wb') as f:
        for i in range(N):
            # Position (3 x float32)
            f.write(struct.pack('fff', 
                positions_world[i, 0], positions_world[i, 1], positions_world[i, 2]))
            # Scales (3 x float32) - direct, NOT log!
            f.write(struct.pack('fff',
                scales_world[i, 0], scales_world[i, 1], scales_world[i, 2]))
            # Color RGBA (4 x uint8)
            f.write(struct.pack('BBBB', 
                colors_r[i], colors_g[i], colors_b[i], colors_a[i]))
            # Rotation quaternion wxyz (4 x uint8)
            f.write(struct.pack('BBBB',
                quat_uint8[i, 0], quat_uint8[i, 1], quat_uint8[i, 2], quat_uint8[i, 3]))
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.1f} KB ({file_size // N} bytes/Gaussian)")
    print(f"  Position range: [{positions_world.min():.3f}, {positions_world.max():.3f}]")
    print(f"  Scale range: [{scales_world.min():.6f}, {scales_world.max():.6f}]")
    print(f"  Intensity range: [{intensities.min():.3f}, {intensities.max():.3f}]")
    print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Export TOPS-GATE model to .splat format')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to checkpoint file (.pt or .pth)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .splat file path')
    parser.add_argument('--scale', '-s', type=float, default=2.0,
                        help='Scale factor for world coordinates (default: 2.0)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        base = os.path.splitext(args.checkpoint)[0]
        args.output = base + '.splat'
    
    # Load parameters
    positions, scales, rotations, intensities, opacities, volume_shape = load_gaussian_params(args.checkpoint)
    
    # Export
    export_to_splat(positions, scales, rotations, intensities, opacities,
                    args.output, scale_factor=args.scale, volume_shape=volume_shape)
    
    print(f"\n✓ Export complete: {args.output}")


if __name__ == '__main__':
    main()
