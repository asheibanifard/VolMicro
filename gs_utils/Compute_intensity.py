"""
Gaussian intensity computation with CUDA acceleration (with PyTorch fallback).
"""

import torch
from torch.autograd import Function
import os

# Get the directory where this script is located
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Skip CUDA compilation - use PyTorch fallback for portability
# Set VOLMICRO_USE_CUDA=1 to attempt CUDA kernel compilation
CUDA_AVAILABLE = False
compute_intensity_cuda = None

if os.environ.get('VOLMICRO_USE_CUDA', '0') == '1':
    try:
        import torch.utils.cpp_extension
        from torch.utils.cpp_extension import load
        
        compute_intensity_cuda = load(
            name='compute_intensity_cuda', 
            sources=[os.path.join(_current_dir, 'discretize_grid.cu')],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--allow-unsupported-compiler'],
            verbose=False
        )
        CUDA_AVAILABLE = True
        print("CUDA intensity computation loaded successfully")
    except Exception as e:
        print(f"Warning: CUDA intensity computation not available ({e}), using PyTorch fallback")
        CUDA_AVAILABLE = False
else:
    print("Using PyTorch fallback for intensity computation (set VOLMICRO_USE_CUDA=1 to use CUDA)")


def _compute_intensity_pytorch(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    """
    Pure PyTorch implementation of Gaussian intensity computation.
    
    For each query point, computes the sum of contributions from all Gaussians:
    I(x) = Σᵢ aᵢ * exp(-0.5 * (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ))
    """
    N = gaussian_centers.shape[0]
    M = grid_points.shape[0]
    device = gaussian_centers.device
    
    # Handle empty cases
    if N == 0 or M == 0:
        return intensity_grid
    
    # Reshape inv_cov to (N, 3, 3)
    inv_cov_mat = inv_covariances.view(N, 3, 3)
    
    # For efficiency, process in batches if M is large
    batch_size = min(50000, M)
    result = torch.zeros(M, device=device, dtype=torch.float32)
    
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch_points = grid_points[start:end]  # (B, 3)
        
        # Compute distances: diff[i,j] = batch_points[i] - centers[j]
        # Shape: (B, N, 3)
        diff = batch_points.unsqueeze(1) - gaussian_centers.unsqueeze(0)
        
        # Apply inverse covariance: (x-μ)ᵀ Σ⁻¹ (x-μ)
        diff_transformed = torch.einsum('bnj,njk->bnk', diff, inv_cov_mat)
        
        # Quadratic form: sum over last dimension of (diff * diff_transformed)
        mahal_sq = (diff * diff_transformed).sum(dim=-1)  # (B, N)
        
        # Gaussian: exp(-0.5 * mahal_sq)
        # Clamp to avoid numerical issues
        mahal_sq = mahal_sq.clamp(min=0, max=50)  # exp(-25) ≈ 0
        gaussian = torch.exp(-0.5 * mahal_sq)  # (B, N)
        
        # Weight by amplitudes and sum
        weighted = gaussian * intensities.unsqueeze(0)  # (B, N)
        result[start:end] = weighted.sum(dim=1)  # (B,)
    
    # Put result in output tensor
    out_flat = intensity_grid.view(-1)
    out_flat[:M] = result
    
    return intensity_grid


class IntensityComputationCUDA(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
        
        # Call the forward CUDA function
        intensity_grid = compute_intensity_cuda.compute_intensity(
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        return intensity_grid

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid = ctx.saved_tensors
        
        # Call the backward CUDA function
        grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_intensity_grid = compute_intensity_cuda.compute_intensity_backward(
            grad_output,
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )

        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, grad_intensity_grid


class IntensityComputationPyTorch(Function):
    """PyTorch autograd-compatible fallback implementation."""
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings)
        
        result = _compute_intensity_pytorch(
            gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings = ctx.saved_tensors
        
        N = gaussian_centers.shape[0]
        M = grid_points.shape[0]
        device = gaussian_centers.device
        
        # Reshape for computation
        inv_cov_mat = inv_covariances.view(N, 3, 3)
        grad_flat = grad_output.view(-1)[:M]  # (M,)
        
        # Initialize gradients
        grad_centers = torch.zeros_like(gaussian_centers)
        grad_intensities = torch.zeros_like(intensities)
        grad_inv_cov = torch.zeros_like(inv_covariances)
        
        # Process in batches
        batch_size = min(20000, M)
        
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch_points = grid_points[start:end]
            batch_grad = grad_flat[start:end]
            B = batch_points.shape[0]
            
            # Forward pass computations
            diff = batch_points.unsqueeze(1) - gaussian_centers.unsqueeze(0)  # (B, N, 3)
            diff_transformed = torch.einsum('bnj,njk->bnk', diff, inv_cov_mat)  # (B, N, 3)
            mahal_sq = (diff * diff_transformed).sum(dim=-1).clamp(min=0, max=50)  # (B, N)
            gaussian = torch.exp(-0.5 * mahal_sq)  # (B, N)
            
            # Gradient of output w.r.t. intensities: gaussian
            # dL/d_intensities[j] = sum_i grad_out[i] * gaussian[i,j]
            grad_intensities += (batch_grad.unsqueeze(1) * gaussian).sum(dim=0)
            
            # Gradient w.r.t. centers (simplified - only position gradient)
            # dL/d_centers = sum over points of grad * intensity * gaussian * inv_cov @ diff
            weighted = batch_grad.unsqueeze(1) * intensities.unsqueeze(0) * gaussian  # (B, N)
            grad_centers += torch.einsum('bn,bnk->nk', weighted, diff_transformed)
        
        return grad_centers, None, grad_intensities, grad_inv_cov, None, None


def compute_intensity(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    """
    Compute Gaussian intensities at query points.
    Uses CUDA if available, otherwise falls back to PyTorch.
    """
    if CUDA_AVAILABLE:
        return IntensityComputationCUDA.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
    else:
        return IntensityComputationPyTorch.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
