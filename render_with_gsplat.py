#!/usr/bin/env python3
"""
Render Gaussian model using diff-gaussian-rasterization library.

This renders the 3D Gaussians from arbitrary camera viewpoints using
the official 3DGS rasterizer.

Usage:
    python render_with_gsplat.py --checkpoint <path_to_checkpoint.pth> --output renders/
"""

import argparse
import os
import sys
import math
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_DIFF_GAUSSIAN = True
except ImportError:
    print("Warning: diff_gaussian_rasterization not found. Install from:")
    print("  cd temp_gs/submodules/diff-gaussian-rasterization && pip install .")
    HAS_DIFF_GAUSSIAN = False


def getWorld2View(R, t):
    """Get world to view matrix. R is 3x3 rotation, t is 3D translation."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T  # Transpose of rotation
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Get OpenGL-style projection matrix."""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)
    P[0, 0] = 1.0 / tanHalfFovX
    P[1, 1] = 1.0 / tanHalfFovY
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def create_camera(azimuth, elevation, distance, target, fov=60, image_size=(512, 512)):
    """
    Create camera parameters for rendering.
    
    Args:
        azimuth: Horizontal angle in degrees
        elevation: Vertical angle in degrees  
        distance: Distance from target
        target: 3D point to look at
        fov: Field of view in degrees
        image_size: (width, height)
    """
    # Convert to radians
    az = math.radians(azimuth)
    el = math.radians(elevation)
    
    # Camera position in spherical coordinates
    x = distance * math.cos(el) * math.sin(az)
    y = distance * math.sin(el)
    z = distance * math.cos(el) * math.cos(az)
    
    camera_pos = np.array([x, y, z], dtype=np.float32) + np.array(target, dtype=np.float32)
    
    # Look at matrix - compute camera orientation
    forward = np.array(target, dtype=np.float32) - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    # World up vector
    world_up = np.array([0, 1, 0], dtype=np.float32)
    
    # Camera right vector
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Camera looking straight up/down, use different up
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    # Camera up vector
    up = np.cross(right, forward)
    
    # Rotation matrix - rows are camera axes directions
    # This transforms world coords to camera coords
    R = np.stack([right, up, -forward], axis=0).astype(np.float32)  # [3, 3]
    
    # Translation in camera space
    t = R @ camera_pos
    
    # World to view 4x4 matrix (row-major)
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    
    # Compute transforms
    W, H = image_size
    fovY = fov * math.pi / 180
    fovX = fovY * W / H  # Adjust for aspect ratio
    
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    znear, zfar = 0.01, 100.0
    
    # Projection matrix (row-major)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tanHalfFovX
    P[1, 1] = 1.0 / tanHalfFovY
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    
    # Convert to column-major for CUDA (transpose)
    world_view_tensor = torch.tensor(Rt).float().cuda().T
    projection_tensor = torch.tensor(P).float().cuda().T
    
    # Full projection
    full_proj = world_view_tensor @ projection_tensor
    
    # Camera center from inverse of world_view
    camera_center = torch.inverse(world_view_tensor)[3, :3]
    
    return {
        'camera_center': camera_center,
        'world_view_transform': world_view_tensor,
        'full_proj_transform': full_proj,
        'image_width': W,
        'image_height': H,
        'tanfovx': tanHalfFovX,
        'tanfovy': tanHalfFovY,
    }


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=-2)
    
    return R


def compute_cov3d(scales, rotations):
    """
    Compute 3D covariance matrices from scales and rotations.
    
    Args:
        scales: [N, 3] scale values
        rotations: [N, 4] quaternions (w, x, y, z)
    
    Returns:
        cov3d: [N, 6] upper triangular covariance (xx, xy, xz, yy, yz, zz)
    """
    # Build rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(rotations)  # [N, 3, 3]
    
    # Build scale matrix
    S = torch.diag_embed(scales)  # [N, 3, 3]
    
    # Covariance = R @ S @ S^T @ R^T
    M = R @ S
    cov = M @ M.transpose(-1, -2)  # [N, 3, 3]
    
    # Extract upper triangular (6 values)
    cov3d = torch.stack([
        cov[:, 0, 0],  # xx
        cov[:, 0, 1],  # xy  
        cov[:, 0, 2],  # xz
        cov[:, 1, 1],  # yy
        cov[:, 1, 2],  # yz
        cov[:, 2, 2],  # zz
    ], dim=-1)
    
    return cov3d


def render_gaussians_diff(means3D, scales, rotations, opacities, colors, camera, 
                          bg_color=torch.tensor([0, 0, 0]), device='cuda'):
    """
    Render Gaussians using diff-gaussian-rasterization.
    
    Args:
        means3D: [N, 3] Gaussian centers in world coordinates
        scales: [N, 3] Gaussian scales
        rotations: [N, 4] Gaussian rotations (quaternion w,x,y,z)
        opacities: [N, 1] Gaussian opacities
        colors: [N, 3] Gaussian colors (RGB)
        camera: Camera dict from create_camera
        bg_color: Background color
        device: CUDA device
    """
    if not HAS_DIFF_GAUSSIAN:
        raise ImportError("diff_gaussian_rasterization not available")
    
    # Move to device and ensure float32
    means3D = means3D.to(device).float().contiguous()
    scales = scales.to(device).float().contiguous()
    rotations = rotations.to(device).float().contiguous()
    opacities = opacities.to(device).float().contiguous()
    colors = colors.to(device).float().contiguous()
    bg_color = bg_color.to(device).float()
    
    N = means3D.shape[0]
    
    # Compute covariance
    cov3d = compute_cov3d(scales, rotations)
    
    # Setup rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=camera['image_height'],
        image_width=camera['image_width'],
        tanfovx=camera['tanfovx'],
        tanfovy=camera['tanfovy'],
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera['world_view_transform'].to(device),
        projmatrix=camera['full_proj_transform'].to(device),
        sh_degree=0,  # No SH, direct colors
        campos=camera['camera_center'].to(device),
        prefiltered=False,
        debug=False,
        antialiasing=True
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Render with precomputed colors (not SHs)
    result = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D[:, :2]),  # Screenspace points
        shs=None,
        colors_precomp=colors,  # Use precomputed colors directly
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    # Handle different return formats
    if isinstance(result, tuple):
        rendered_image = result[0]
        radii = result[1] if len(result) > 1 else None
    else:
        rendered_image = result
        radii = None
    
    return rendered_image, radii


def render_gaussians_simple(means3D, scales, opacities, colors, camera, 
                            image_size=(512, 512), device='cuda'):
    """
    Simple software renderer for Gaussians (fallback when diff-gaussian unavailable).
    
    Projects Gaussians to 2D and renders using alpha blending.
    """
    W, H = image_size
    
    # Create output image
    image = torch.zeros(3, H, W, device=device)
    depth = torch.full((H, W), float('inf'), device=device)
    
    # Get camera matrices
    world_view = camera['world_view_transform'].to(device)
    proj = camera['projection_matrix'].to(device)
    
    # Transform to camera space
    ones = torch.ones(means3D.shape[0], 1, device=device)
    means_hom = torch.cat([means3D.to(device), ones], dim=1)  # [N, 4]
    
    # World to view
    means_view = means_hom @ world_view  # [N, 4]
    
    # Project
    means_proj = means_view @ proj  # [N, 4]
    means_ndc = means_proj[:, :3] / (means_proj[:, 3:4] + 1e-8)  # [N, 3]
    
    # NDC to pixel coords
    x_pix = ((means_ndc[:, 0] + 1) * 0.5 * W).long()
    y_pix = ((1 - means_ndc[:, 1]) * 0.5 * H).long()  # Flip Y
    z_depth = means_view[:, 2]
    
    # Filter visible points
    valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H) & (z_depth > 0)
    
    # Sort by depth (back to front)
    sort_idx = torch.argsort(z_depth, descending=True)
    
    # Simple splatting
    colors = colors.to(device)
    opacities = opacities.to(device).squeeze()
    scales = scales.to(device)
    
    for idx in sort_idx:
        if not valid[idx]:
            continue
            
        x, y = x_pix[idx].item(), y_pix[idx].item()
        scale = scales[idx].mean().item() * min(W, H) * 2  # Approximate splat size
        alpha = opacities[idx].item()
        color = colors[idx]
        
        # Draw gaussian splat (simple circle approximation)
        radius = max(1, int(scale * 3))
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if 0 <= px < W and 0 <= py < H:
                    dist_sq = (dx * dx + dy * dy) / (scale * scale + 1e-8)
                    if dist_sq < 9:  # 3 sigma
                        weight = alpha * math.exp(-0.5 * dist_sq)
                        image[:, py, px] = image[:, py, px] * (1 - weight) + color * weight
    
    return image.clamp(0, 1)


def load_gaussians_from_checkpoint(checkpoint_path, device='cuda'):
    """Load Gaussian parameters from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        positions = state['positions']
        scales = state.get('scales', state.get('log_scales', None))
        # Handle different opacity/intensity naming
        opacities = state.get('opacities', state.get('raw_opacities', state.get('intensities', None)))
        rotations = state.get('rotations', None)
        
        if scales is not None and scales.min() < -1:  # Log scale (very negative values)
            scales = torch.exp(scales)
        if opacities is not None:
            # Check if needs sigmoid activation
            if opacities.min() < 0 or opacities.max() > 1:
                opacities = torch.sigmoid(opacities)
            
    elif 'xyz' in checkpoint:
        # 3DGR-CT format
        positions = checkpoint['xyz']
        scales = torch.exp(checkpoint['scaling'])  # Log scale -> scale
        opacities = torch.sigmoid(checkpoint['intensity'])  # Raw -> sigmoid
        rotations = checkpoint['rotation']
        
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")
    
    # Get volume shape for reference
    if 'model_params' in checkpoint:
        vol_shape = checkpoint['model_params']['volume_shape']
    elif 'config' in checkpoint:
        vol_shape = checkpoint['config'].get('img_size', [100, 100, 100])
    else:
        vol_shape = [100, 100, 100]
    
    vol_shape = np.array(vol_shape)
    
    print(f"  Loaded {positions.shape[0]} Gaussians")
    print(f"  Volume shape: {vol_shape}")
    print(f"  Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    
    # Use REAL dimensions - positions are in [0,1] normalized coords
    # Convert to real voxel coordinates centered at origin
    # vol_shape is (D, H, W) = (100, 647, 813)
    D, H, W = vol_shape
    max_dim = max(D, H, W)
    
    # Scale positions to real dimensions, centered at origin
    # positions are [N, 3] with x, y, z in [0, 1]
    positions_real = positions.float().clone()
    positions_real[:, 0] = (positions[:, 0] - 0.5) * W / max_dim  # x
    positions_real[:, 1] = (positions[:, 1] - 0.5) * H / max_dim  # y  
    positions_real[:, 2] = (positions[:, 2] - 0.5) * D / max_dim  # z
    
    # Scale the Gaussians to real dimensions
    scales_real = scales.float().clone()
    scales_real[:, 0] = scales[:, 0] * W / max_dim
    scales_real[:, 1] = scales[:, 1] * H / max_dim
    scales_real[:, 2] = scales[:, 2] * D / max_dim
    
    print(f"  Real position range: [{positions_real.min():.3f}, {positions_real.max():.3f}]")
    print(f"  Real scale range: [{scales_real.min():.4f}, {scales_real.max():.4f}]")
    
    # Ensure proper shapes
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(-1)
    
    # Use existing rotations or create identity
    N = positions.shape[0]
    if rotations is None:
        rotations = torch.zeros(N, 4, device=device)
        rotations[:, 0] = 1.0  # w = 1 for identity quaternion
    else:
        rotations = rotations.float()
        # Normalize quaternions
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Colors - white, let opacity control brightness
    colors = torch.ones(N, 3, device=device)
    
    # Clamp opacity to valid range and use as-is
    opacities = torch.clamp(opacities, 0.01, 1.0)
    
    print(f"  Final opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    print(f"  Final scale range: [{scales_real.min():.4f}, {scales_real.max():.4f}]")
    
    return {
        'positions': positions_real.to(device),
        'scales': scales_real.to(device),
        'rotations': rotations.to(device),
        'opacities': opacities.to(device),
        'colors': colors.to(device),
        'volume_shape': vol_shape,
    }


def render_orbit(gaussians, output_dir, num_frames=36, elevation=30, distance=3.0,
                 image_size=(512, 512), device='cuda'):
    """Render orbit around the Gaussians."""
    os.makedirs(output_dir, exist_ok=True)
    
    target = [0, 0, 0]  # Look at origin
    
    images = []
    
    for i in range(num_frames):
        azimuth = i * 360 / num_frames
        
        camera = create_camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            target=target,
            fov=60,
            image_size=image_size
        )
        
        # Render
        if HAS_DIFF_GAUSSIAN:
            image, _ = render_gaussians_diff(
                gaussians['positions'],
                gaussians['scales'],
                gaussians['rotations'],
                gaussians['opacities'],
                gaussians['colors'],
                camera,
                device=device
            )
        else:
            image = render_gaussians_simple(
                gaussians['positions'],
                gaussians['scales'],
                gaussians['opacities'],
                gaussians['colors'],
                camera,
                image_size=image_size,
                device=device
            )
        
        # Save frame
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        img_pil.save(frame_path)
        images.append(img_pil)
        
        print(f"\rRendering frame {i+1}/{num_frames}", end='')
    
    print()
    
    # Save as GIF
    gif_path = os.path.join(output_dir, 'orbit.gif')
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    print(f"Saved GIF to: {gif_path}")
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Render Gaussians using diff-gaussian-rasterization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='renders',
                        help='Output directory for rendered images')
    parser.add_argument('--num_frames', type=int, default=36,
                        help='Number of frames for orbit')
    parser.add_argument('--elevation', type=float, default=30,
                        help='Camera elevation angle in degrees')
    parser.add_argument('--distance', type=float, default=3.0,
                        help='Camera distance from center')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Output image size (width height)')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load Gaussians
    gaussians = load_gaussians_from_checkpoint(args.checkpoint, args.device)
    
    # Render orbit
    print(f"\nRendering {args.num_frames} frames...")
    print(f"  Using {'diff-gaussian-rasterization' if HAS_DIFF_GAUSSIAN else 'simple software renderer'}")
    
    render_orbit(
        gaussians,
        args.output,
        num_frames=args.num_frames,
        elevation=args.elevation,
        distance=args.distance,
        image_size=tuple(args.image_size),
        device=args.device
    )
    
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == '__main__':
    main()
