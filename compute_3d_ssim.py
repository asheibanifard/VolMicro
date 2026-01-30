#!/usr/bin/env python3
"""
Compute 3D-SSIM for VolMicro reconstruction.

Supports both full-volume and gated (masked) SSIM computation.
"""

import numpy as np
import tifffile
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import argparse


def compute_3d_ssim(gt: np.ndarray, recon: np.ndarray, 
                    mask: np.ndarray = None, win_size: int = 7) -> dict:
    """
    Compute 3D-SSIM between ground truth and reconstruction.
    
    Args:
        gt: Ground truth volume (D, H, W)
        recon: Reconstructed volume (D, H, W)
        mask: Optional gating mask (D, H, W), values in [0, 1] or [0, 255]
        win_size: SSIM window size (default: 7)
        
    Returns:
        Dictionary with SSIM metrics
    """
    # Normalize volumes to [0, 1]
    gt_min, gt_max = gt.min(), gt.max()
    gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)
    recon_norm = np.clip((recon - gt_min) / (gt_max - gt_min + 1e-8), 0, 1)
    
    # Compute full 3D-SSIM with SSIM map
    ssim_full, ssim_map = structural_similarity(
        gt_norm, recon_norm, 
        data_range=1.0, 
        win_size=win_size, 
        full=True
    )
    
    results = {
        'ssim_full': ssim_full,
        'ssim_map': ssim_map,
    }
    
    # If mask provided, compute gated SSIM
    if mask is not None:
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask_f = mask.astype(np.float32) / 255.0
        else:
            mask_f = mask.astype(np.float32)
        
        # Weighted SSIM (using mask as weights)
        mask_sum = mask_f.sum()
        if mask_sum > 0:
            ssim_gated = np.sum(ssim_map * mask_f) / mask_sum
        else:
            ssim_gated = ssim_full
        
        results['ssim_gated'] = ssim_gated
        results['n_gated_voxels'] = int((mask_f > 0.5).sum())
        results['gated_fraction'] = results['n_gated_voxels'] / mask.size
        
        # Per-slice gated SSIM
        ssim_slices = []
        mask_bool = mask_f > 0.5
        for z in range(gt.shape[0]):
            if mask_bool[z].sum() > 100:
                _, smap = structural_similarity(
                    gt_norm[z], recon_norm[z], 
                    data_range=1.0, 
                    full=True
                )
                m = mask_f[z]
                if m.sum() > 0:
                    ssim_slices.append(np.sum(smap * m) / np.sum(m))
        
        if ssim_slices:
            results['ssim_slice_mean'] = np.mean(ssim_slices)
            results['ssim_slice_std'] = np.std(ssim_slices)
    
    return results


def compute_3d_psnr(gt: np.ndarray, recon: np.ndarray, 
                    mask: np.ndarray = None) -> dict:
    """
    Compute 3D-PSNR between ground truth and reconstruction.
    
    Args:
        gt: Ground truth volume (D, H, W)
        recon: Reconstructed volume (D, H, W)
        mask: Optional gating mask (D, H, W)
        
    Returns:
        Dictionary with PSNR metrics
    """
    data_range = gt.max() - gt.min()
    
    # Full volume PSNR
    psnr_full = peak_signal_noise_ratio(gt, recon, data_range=data_range)
    
    results = {'psnr_full': psnr_full}
    
    # Gated PSNR
    if mask is not None:
        if mask.max() > 1:
            mask_bool = mask > 127
        else:
            mask_bool = mask > 0.5
        
        gt_gated = gt[mask_bool]
        recon_gated = recon[mask_bool]
        
        mse_gated = np.mean((gt_gated - recon_gated) ** 2)
        if mse_gated > 0:
            # PSNR with data_range=1.0 (normalized volume)
            psnr_gated = 10 * np.log10(1.0 / mse_gated)
        else:
            psnr_gated = float('inf')
        
        results['psnr_gated'] = psnr_gated
        results['mse_gated'] = mse_gated
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute 3D-SSIM for VolMicro')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth TIFF')
    parser.add_argument('--recon', type=str, required=True, help='Reconstruction TIFF')
    parser.add_argument('--mask', type=str, default=None, help='Gate mask TIFF (optional)')
    parser.add_argument('--win-size', type=int, default=7, help='SSIM window size')
    args = parser.parse_args()
    
    # Load volumes
    print(f'Loading ground truth: {args.gt}')
    gt = tifffile.imread(args.gt)
    
    print(f'Loading reconstruction: {args.recon}')
    recon = tifffile.imread(args.recon)
    
    mask = None
    if args.mask:
        print(f'Loading mask: {args.mask}')
        mask = tifffile.imread(args.mask)
    
    print(f'\nVolume shape: {gt.shape}')
    print(f'GT range: [{gt.min():.4f}, {gt.max():.4f}]')
    print(f'Recon range: [{recon.min():.4f}, {recon.max():.4f}]')
    
    # Compute metrics
    print('\nComputing 3D-SSIM...')
    ssim_results = compute_3d_ssim(gt, recon, mask, args.win_size)
    
    print('Computing 3D-PSNR...')
    psnr_results = compute_3d_psnr(gt, recon, mask)
    
    # Print results
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    
    print(f'\n3D-SSIM (full volume):    {ssim_results["ssim_full"]:.4f}')
    print(f'PSNR (full volume):       {psnr_results["psnr_full"]:.2f} dB')
    
    if mask is not None:
        print(f'\n3D-SSIM (gated):          {ssim_results["ssim_gated"]:.4f}')
        print(f'PSNR (gated):             {psnr_results["psnr_gated"]:.2f} dB')
        print(f'Gated voxels:             {ssim_results["n_gated_voxels"]:,} ({100*ssim_results["gated_fraction"]:.1f}%)')
        
        if 'ssim_slice_mean' in ssim_results:
            print(f'Slice-averaged SSIM:      {ssim_results["ssim_slice_mean"]:.4f} ± {ssim_results["ssim_slice_std"]:.4f}')
    
    print('=' * 60)
    
    return ssim_results, psnr_results


if __name__ == '__main__':
    main()
