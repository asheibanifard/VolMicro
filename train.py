"""
Train Gaussian-Based Volume Data Representation

Implementation based on [21 Dec. 24] Algorithm from research proposal.

This script trains the Gaussian volume model to represent volume data
using an implicit neural representation based on Gaussian basis functions.

Summary from document:
    min_{N, u_i, Σ_i, w_i} Σ_{k=1}^{M} ||v_k(x_k, y_k, z_k) - Σ_{i=1}^{N} w_i * G_i(x_k, y_k, z_k; u_i, Σ_i)||^2

Usage:
    python train.py --config config.yaml
    python train.py --volume path/to/volume.tif --num_gaussians 5000
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dec24_GaussianVolume import (
    GaussianVolumeModel,
    GaussianVolumeTrainer,
    create_trainer,
    compute_psnr
)


def load_volume(path: str, normalize: bool = True) -> torch.Tensor:
    """
    Load volume data from file.
    
    Args:
        path: Path to volume file (TIFF, NPY, etc.)
        normalize: Whether to normalize values to [0, 1]
        
    Returns:
        Volume tensor of shape (D, H, W)
    """
    path = Path(path)
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        volume = tifffile.imread(str(path))
    elif path.suffix.lower() == '.npy':
        volume = np.load(str(path))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    volume = volume.astype(np.float32)
    
    if normalize:
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
    
    return torch.from_numpy(volume)


def save_results(
    model: GaussianVolumeModel,
    trainer: GaussianVolumeTrainer,
    output_dir: str,
    original_volume: torch.Tensor
):
    """
    Save training results and visualizations.
    
    Args:
        model: Trained model
        trainer: Trainer with history
        output_dir: Output directory
        original_volume: Original volume for comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].semilogy(trainer.history['total_loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    # MSE loss
    axes[0, 1].semilogy(trainer.history['mse_loss'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].grid(True)
    
    # PSNR
    if trainer.history['psnr']:
        axes[1, 0].plot(trainer.history['psnr'])
        axes[1, 0].set_xlabel('Evaluation')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('PSNR')
        axes[1, 0].grid(True)
    
    # Regularization losses
    ax = axes[1, 1]
    if trainer.history['sparsity_loss']:
        ax.semilogy(trainer.history['sparsity_loss'], label='Sparsity')
    if trainer.history['overlap_loss']:
        ax.semilogy(trainer.history['overlap_loss'], label='Overlap')
    if trainer.history['smoothness_loss']:
        ax.semilogy(trainer.history['smoothness_loss'], label='Smoothness')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Regularization Losses')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Reconstruct volume and compare
    print("Reconstructing volume...")
    reconstructed = model.reconstruct_volume()
    
    # Save comparison slices
    D, H, W = original_volume.shape
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Original slices
    axes[0, 0].imshow(original_volume[mid_d].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original (XY)')
    axes[0, 1].imshow(original_volume[:, mid_h].cpu().numpy(), cmap='gray')
    axes[0, 1].set_title('Original (XZ)')
    axes[0, 2].imshow(original_volume[:, :, mid_w].cpu().numpy(), cmap='gray')
    axes[0, 2].set_title('Original (YZ)')
    
    # Reconstructed slices
    axes[1, 0].imshow(reconstructed[mid_d].cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Reconstructed (XY)')
    axes[1, 1].imshow(reconstructed[:, mid_h].cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Reconstructed (XZ)')
    axes[1, 2].imshow(reconstructed[:, :, mid_w].cpu().numpy(), cmap='gray')
    axes[1, 2].set_title('Reconstructed (YZ)')
    
    # Difference
    diff = (original_volume - reconstructed).abs()
    axes[2, 0].imshow(diff[mid_d].cpu().numpy(), cmap='hot')
    axes[2, 0].set_title('Difference (XY)')
    axes[2, 1].imshow(diff[:, mid_h].cpu().numpy(), cmap='hot')
    axes[2, 1].set_title('Difference (XZ)')
    axes[2, 2].imshow(diff[:, :, mid_w].cpu().numpy(), cmap='hot')
    axes[2, 2].set_title('Difference (YZ)')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_comparison.png'), dpi=150)
    plt.close()
    
    # Save reconstructed volume
    recon_np = reconstructed.cpu().numpy()
    tifffile.imwrite(os.path.join(output_dir, 'reconstructed_volume.tif'), recon_np)
    
    # Save Gaussian parameters
    params = model.gaussians.get_parameters_dict()
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            params[key] = value.cpu().numpy()
    np.savez(os.path.join(output_dir, 'gaussian_parameters.npz'), **params)
    
    # Compute final metrics
    psnr = compute_psnr(reconstructed.flatten(), original_volume.flatten().to(reconstructed.device))
    mse = torch.nn.functional.mse_loss(reconstructed, original_volume.to(reconstructed.device)).item()
    
    metrics = {
        'final_psnr': psnr,
        'final_mse': mse,
        'num_gaussians': model.num_gaussians,
        'compression_ratio': (original_volume.numel() * 4) / (model.num_gaussians * (3 + 3 + 4 + 1) * 4)
    }
    
    with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f)
    
    print(f"\nFinal Metrics:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MSE: {mse:.6f}")
    print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Train Gaussian-Based Volume Data Representation'
    )
    
    # Data
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to volume data (TIFF or NPY)')
    
    # Model
    parser.add_argument('--num_gaussians', type=int, default=5000,
                        help='Number of Gaussian basis functions (N)')
    parser.add_argument('--init_method', type=str, default='uniform',
                        choices=['uniform', 'grid'],
                        help='Gaussian initialization method')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (ρ)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    
    # Regularization (Step 6)
    parser.add_argument('--lambda_sparsity', type=float, default=0.001,
                        help='Sparsity regularization weight (λ_w)')
    parser.add_argument('--lambda_overlap', type=float, default=0.001,
                        help='Overlap regularization weight (λ_o)')
    parser.add_argument('--lambda_smoothness', type=float, default=0.001,
                        help='Smoothness regularization weight (λ_s)')
    parser.add_argument('--no_sparsity', action='store_true',
                        help='Disable sparsity regularization')
    parser.add_argument('--no_overlap', action='store_true',
                        help='Disable overlap regularization')
    parser.add_argument('--no_smoothness', action='store_true',
                        help='Disable smoothness regularization')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Checkpoint save interval (epochs)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Gaussian-Based Volume Data Representation")
    print("Implementation based on [21 Dec. 24] Algorithm")
    print("=" * 60)
    
    # Load volume data
    print(f"\nLoading volume from: {args.volume}")
    volume = load_volume(args.volume)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Value range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    # Create model and trainer
    print(f"\nInitializing model with {args.num_gaussians} Gaussians...")
    model, trainer = create_trainer(
        volume=volume,
        num_gaussians=args.num_gaussians,
        learning_rate=args.lr,
        optimizer_type=args.optimizer,
        lambda_sparsity=args.lambda_sparsity if not args.no_sparsity else 0,
        lambda_overlap=args.lambda_overlap if not args.no_overlap else 0,
        lambda_smoothness=args.lambda_smoothness if not args.no_smoothness else 0,
        init_method=args.init_method,
        device=args.device
    )
    
    # Print training configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  λ_sparsity: {args.lambda_sparsity if not args.no_sparsity else 'disabled'}")
    print(f"  λ_overlap: {args.lambda_overlap if not args.no_overlap else 'disabled'}")
    print(f"  λ_smoothness: {args.lambda_smoothness if not args.no_smoothness else 'disabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    # Save configuration
    config = vars(args)
    config['volume_shape'] = list(volume.shape)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Train
    print(f"\nStarting training...")
    history = trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=checkpoint_dir,
        verbose=True
    )
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    save_results(model, trainer, args.output_dir, volume)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
