"""
Loss Functions for Gaussian-Based Volume Data Representation
=============================================================

Implementation based on [21 Dec. 24] Algorithm from research proposal.

Mathematical Framework
----------------------
We represent a 3D volume V(x,y,z) as a mixture of N anisotropic Gaussians:

    V(x,y,z) = Σᵢ₌₁ᴺ wᵢ · G(x,y,z; μᵢ, Σᵢ)

where each Gaussian is defined as:

    G(x,y,z; μ, Σ) = exp(-½ (p - μ)ᵀ Σ⁻¹ (p - μ))

with:
    - p = [x, y, z]ᵀ : query point
    - μᵢ ∈ ℝ³ : center position of Gaussian i
    - Σᵢ ∈ ℝ³ˣ³ : covariance matrix (positive semi-definite)
    - wᵢ ∈ ℝ : intensity/weight of Gaussian i

The covariance matrix is parameterized as:
    
    Σ = R · S · Sᵀ · Rᵀ

where:
    - R ∈ SO(3) : rotation matrix (from quaternion q ∈ ℝ⁴, ||q|| = 1)
    - S = diag(s₁, s₂, s₃) : scale matrix with sⱼ > 0

Loss Functions
--------------
1. MSE Loss (L_mse) - Main reconstruction loss
2. Sparsity Regularization (L_sparse) - L1 penalty on weights
3. Overlap Regularization (L_overlap) - Penalize Gaussian overlap
4. Smoothness Regularization (L_smooth) - Encourage smooth parameter fields
5. Total Loss: L_total = L_mse + λ_s·L_sparse + λ_o·L_overlap + λ_sm·L_smooth

References
----------
[1] Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 
    SIGGRAPH 2023
[2] Zwicker et al., "EWA Splatting", IEEE TVCG 2002
[3] Yu et al., "Mip-Splatting: Alias-free 3D Gaussian Splatting", CVPR 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import FAISS for GPU-accelerated KNN in overlap loss
try:
    import faiss
    import faiss.contrib.torch_utils
    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False


class ReconstructionLoss(nn.Module):
    """
    Mean Squared Error Loss for volume reconstruction.
    
    Mathematical Formulation
    ------------------------
    Given M sampled voxel positions and their ground truth values, the MSE loss is:
    
        L_mse = (1/M) · Σₖ₌₁ᴹ (f(xₖ, yₖ, zₖ) - vₖ)²
    
    where:
        - M : number of sampled voxels (can be full volume or random subset)
        - (xₖ, yₖ, zₖ) : 3D coordinates of voxel k
        - f(·) : predicted intensity from Gaussian mixture
        - vₖ : ground truth voxel intensity
    
    The predicted value at each point is computed as:
    
        f(x,y,z) = Σᵢ₌₁ᴺ wᵢ · αᵢ · exp(-½ dᵢᵀ Σᵢ⁻¹ dᵢ)
    
    where dᵢ = [x,y,z]ᵀ - μᵢ is the displacement from Gaussian center i,
    wᵢ is the intensity, and αᵢ ∈ [0,1] is the opacity.
    
    Gradient w.r.t. Parameters
    --------------------------
    For position μᵢ:
        ∂L/∂μᵢ = (2/M) · Σₖ (f(pₖ) - vₖ) · wᵢαᵢ · Gᵢ(pₖ) · Σᵢ⁻¹ · dᵢₖ
    
    For scale sⱼ (via log-scale for numerical stability):
        ∂L/∂log(sⱼ) = sⱼ · ∂L/∂sⱼ
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and ground truth values.
        
        Args:
            predicted: Predicted voxel values of shape (M,) or (D, H, W)
            ground_truth: Ground truth voxel values of shape (M,) or (D, H, W)
            
        Returns:
            MSE loss value (scalar tensor)
            
        Notes:
            - For full-volume training: M = D × H × W
            - For sampled training: M = num_samples (typically 0.1% - 1% of volume)
            - Loss is normalized by M, making it comparable across different sample sizes
        """
        return self.mse(predicted, ground_truth)


class SparsityRegularization(nn.Module):
    """
    Weight Sparsity Regularization (L1 Penalty).
    
    Mathematical Formulation
    ------------------------
    The L1 sparsity regularization encourages sparse weight distributions:
    
        L_sparse = λ_w · Σᵢ₌₁ᴺ |wᵢ|
    
    where:
        - N : number of Gaussians
        - wᵢ : intensity/weight of Gaussian i
        - λ_w : regularization coefficient (hyperparameter)
    
    Motivation
    ----------
    L1 regularization promotes sparsity by penalizing the absolute magnitude
    of weights. This encourages the model to:
    
    1. Use fewer active Gaussians (many weights → 0)
    2. Represent the volume with minimal complexity
    3. Improve generalization by preventing overfitting
    
    The L1 penalty is non-differentiable at w=0, but subgradient methods
    (used by optimizers like Adam) handle this effectively.
    
    Gradient
    --------
        ∂L_sparse/∂wᵢ = λ_w · sign(wᵢ)
    
    where sign(x) = +1 if x > 0, -1 if x < 0, and ∈ [-1,1] if x = 0.
    
    Hyperparameter Selection
    ------------------------
    - λ_w ∈ [1e-4, 1e-2] typical range
    - Higher values → sparser solution, potentially underfitting
    - Lower values → denser solution, better reconstruction
    """
    
    def __init__(self, lambda_w: float = 0.01):
        """
        Args:
            lambda_w: Regularization coefficient (default: 0.01)
        """
        super().__init__()
        self.lambda_w = lambda_w
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity regularization.
        
        L_sparsity = λ_w · (1/N) · Σᵢ₌₁ᴺ |wᵢ|
        
        Note: We normalize by N to keep the loss scale independent of
        the number of Gaussians, making lambda values more intuitive.
        The formula still uses sum (matching proposal) but is normalized.
        
        Args:
            weights: Gaussian weights/intensities of shape (N,)
            
        Returns:
            Sparsity loss value: λ_w · mean(|wᵢ|)
        """
        N = weights.shape[0]
        return self.lambda_w * torch.sum(torch.abs(weights)) / N


class OverlapRegularization(nn.Module):
    """
    Overlap Regularization to prevent Gaussian redundancy.
    
    Mathematical Formulation
    ------------------------
    The overlap regularization penalizes excessive overlap between Gaussians:
    
        L_overlap = λ_o · Σᵢ₌₁ᴺ Σⱼ>ᵢ O(Gᵢ, Gⱼ)
    
    where O(Gᵢ, Gⱼ) measures the overlap between Gaussians i and j.
    
    Overlap Metrics
    ---------------
    Several metrics can quantify Gaussian overlap:
    
    1. **Bhattacharyya Coefficient** (exact, expensive):
        BC(Gᵢ, Gⱼ) = ∫ √(Gᵢ(x) · Gⱼ(x)) dx
        
        For two Gaussians with means μᵢ, μⱼ and covariances Σᵢ, Σⱼ:
        BC = exp(-¼ dᵀ Σ⁻¹ d) · |Σ|^(1/4) / (|Σᵢ|^(1/8) · |Σⱼ|^(1/8))
        
        where Σ = (Σᵢ + Σⱼ)/2 and d = μᵢ - μⱼ
    
    2. **Simplified Distance-Based** (used here, efficient):
        O(Gᵢ, Gⱼ) = exp(-||μᵢ - μⱼ||² / (2(rᵢ + rⱼ)²))
        
        where rᵢ = √(tr(Σᵢ)/3) is the effective radius of Gaussian i.
    
    This approximation:
    - Approaches 1 when Gaussians are coincident
    - Decays exponentially with separation
    - Is cheap to compute (O(N²) distance matrix)
    
    Motivation
    ----------
    Without overlap regularization, Gaussians may:
    - Pile up in high-intensity regions
    - Create redundant representations
    - Waste model capacity
    
    With overlap penalty:
    - Gaussians spread to cover volume efficiently
    - Each Gaussian contributes unique information
    - Better utilization of limited Gaussian budget
    
    Computational Complexity
    ------------------------
    - Naive: O(N²) for all pairs
    - With max_pairs sampling: O(max_pairs) 
    - GPU-accelerated distance matrix is efficient for N < 100k
    """
    
    def __init__(self, lambda_o: float = 0.01):
        """
        Args:
            lambda_o: Regularization coefficient (default: 0.01)
        """
        super().__init__()
        self.lambda_o = lambda_o
    
    def forward(
        self,
        positions: torch.Tensor,
        covariance: torch.Tensor,
        knn_k: int = 16,
        max_gaussians: int = 5000
    ) -> torch.Tensor:
        """
        Compute overlap regularization using KNN-based sampling (memory efficient).
        
        Algorithm (memory-efficient O(N*K) instead of O(N²)):
            1. Sample up to max_gaussians if N is large
            2. For each Gaussian, find K nearest neighbors
            3. Compute overlap only with those K neighbors
            4. Sum overlaps (approximation of full pairwise sum)
        
        Args:
            positions: Gaussian centers μᵢ of shape (N, 3)
            covariance: Covariance matrices Σᵢ of shape (N, 3, 3)
            knn_k: Number of nearest neighbors to consider (default: 16)
            max_gaussians: Max Gaussians to sample for very large N (default: 5000)
            
        Returns:
            Overlap loss: λ_o · Σᵢ Σⱼ∈KNN(i) O(Gᵢ, Gⱼ)
        """
        N = positions.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Sample Gaussians if N is too large
        if N > max_gaussians:
            idx = torch.randperm(N, device=positions.device)[:max_gaussians]
            positions = positions[idx]
            covariance = covariance[idx]
            N = max_gaussians
        
        # Compute effective radius: r = √(tr(Σ)/3)
        scales = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2).mean(dim=1))  # (N,)
        
        # For speed: sample a small subset of Gaussians for overlap computation
        # Full overlap on all N is O(N²) which is too slow for training
        sample_size = min(1000, N)  # Only use 1000 Gaussians max
        if N > sample_size:
            idx = torch.randperm(N, device=positions.device)[:sample_size]
            positions = positions[idx]
            scales = scales[idx]
            N = sample_size
        
        # Adjust K to not exceed N-1
        K = min(knn_k, N - 1)
        
        # Single vectorized computation (N is small now, ~1000)
        # Pairwise distances: (N, N) - only 1M elements for N=1000
        dist_sq = torch.cdist(positions, positions, p=2).pow(2)  # (N, N)
        
        # Combined scales for all pairs: (N, N)
        combined_r = scales.unsqueeze(1) + scales.unsqueeze(0)  # (N, N)
        
        # Overlap matrix: (N, N)
        overlap = torch.exp(-dist_sq / (2 * combined_r ** 2 + 1e-6))
        
        # Zero diagonal (self-overlap) - use mask instead of inplace
        diag_mask = torch.eye(N, dtype=torch.bool, device=positions.device)
        overlap = overlap.masked_fill(diag_mask, 0.0)
        
        # Sum upper triangle only (each pair once)
        # Normalize by number of pairs: N*(N-1)/2
        num_pairs = N * (N - 1) / 2
        total_overlap = overlap.triu(diagonal=1).sum() / max(1.0, num_pairs)
        
        return self.lambda_o * total_overlap


class SmoothnessRegularization(nn.Module):
    """
    Smoothness Regularization for spatially coherent Gaussian fields.
    
    Mathematical Formulation
    ------------------------
    The smoothness regularization encourages nearby Gaussians to have 
    similar parameters, creating a spatially coherent representation:
    
        L_smooth = λ_s · Σᵢ₌₁ᴺ Σⱼ∈𝒩(i) [ (wᵢ - wⱼ)² + ||log(sᵢ) - log(sⱼ)||² ]
    
    where:
        - 𝒩(i) : k-nearest neighbors of Gaussian i (by position)
        - wᵢ : intensity/weight of Gaussian i  
        - sᵢ : scale vector of Gaussian i
        - λ_s : regularization coefficient
    
    Why Log-Scale?
    --------------
    We regularize log(s) instead of s directly because:
    
    1. **Scale invariance**: Penalizing (s₁ - s₂)² treats a change from 
       0.01→0.02 differently than 0.1→0.2, though both are 2× changes.
       Using log: (log(0.02)-log(0.01))² = (log(0.2)-log(0.1))² = (log 2)²
    
    2. **Numerical stability**: Scales span orders of magnitude; 
       log compression prevents large scales from dominating.
    
    3. **Multiplicative regularization**: log-difference corresponds to
       ratio regularization: ||log(sᵢ/sⱼ)||² penalizes scale ratios.
    
    k-Nearest Neighbors
    -------------------
    We use spatial neighbors (by μᵢ position) rather than all pairs because:
    
    1. Distant Gaussians should be independent
    2. O(N·k) complexity vs O(N²) for all pairs
    3. Aligns with physical intuition: nearby regions should be similar
    
    Gradient
    --------
    For weight smoothness:
        ∂L_smooth/∂wᵢ = 2λ_s · Σⱼ∈𝒩(i) (wᵢ - wⱼ)
        
    This acts as a graph Laplacian smoothing operator.
    
    Connection to Total Variation
    -----------------------------
    This is related to Total Variation (TV) regularization but uses
    L2 norm instead of L1, making it differentiable everywhere and
    encouraging smooth transitions rather than piecewise constant fields.
    """
    
    def __init__(self, lambda_s: float = 0.01, num_neighbors: int = 5):
        """
        Args:
            lambda_s: Regularization coefficient (default: 0.01)
            num_neighbors: Number of nearest neighbors k (default: 5)
        """
        super().__init__()
        self.lambda_s = lambda_s
        self.num_neighbors = num_neighbors
    
    def forward(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        log_scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness regularization over k-nearest neighbor graph.
        
        Algorithm:
            1. Build kNN graph: find k nearest neighbors for each Gaussian
            2. Compute weight differences: Δwᵢⱼ = wᵢ - wⱼ for j ∈ 𝒩(i)
            3. Compute scale differences: Δsᵢⱼ = log(sᵢ) - log(sⱼ)
            4. Sum squared differences: L = λ·(mean(Δw²) + mean(Δs²))
        
        Args:
            positions: Gaussian centers μᵢ of shape (N, 3)
            weights: Gaussian weights wᵢ of shape (N,)
            log_scales: Log-scales log(sᵢ) of shape (N, 3)
            
        Returns:
            Smoothness loss: λ_s · (L_weight + L_scale)
        """
        N = positions.shape[0]
        
        if N < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Compute pairwise Euclidean distances: D[i,j] = ||μᵢ - μⱼ||
        distances = torch.cdist(positions, positions)  # (N, N)
        
        # Get k nearest neighbors for each Gaussian (excluding self)
        k = min(self.num_neighbors, N - 1)
        _, indices = torch.topk(distances, k + 1, largest=False, dim=1)
        neighbor_indices = indices[:, 1:]  # (N, k) - exclude self (index 0)
        
        # Discrete gradient approximation: ∇_u G_i ≈ (G_i - G_j) / ||μ_i - μ_j||
        # L_smoothness = λ_s · Σᵢ ||∇_u Gᵢ||²
        
        # Get neighbor distances for normalization
        neighbor_distances = distances.gather(1, neighbor_indices)  # (N, k)
        neighbor_distances = neighbor_distances.clamp(min=1e-6)  # Avoid div by zero
        
        # Weight gradient: ∇w ≈ (wᵢ - wⱼ) / d_ij
        weight_neighbors = weights[neighbor_indices]  # (N, k)
        weight_diff = weights.unsqueeze(1) - weight_neighbors  # (N, k)
        weight_grad_sq = (weight_diff / neighbor_distances) ** 2  # (N, k)
        weight_smoothness = torch.sum(weight_grad_sq)  # Σᵢ Σⱼ∈𝒩(i)
        
        # Scale gradient: ∇s ≈ (log sᵢ - log sⱼ) / d_ij  
        scale_neighbors = log_scales[neighbor_indices]  # (N, k, 3)
        scale_diff = log_scales.unsqueeze(1) - scale_neighbors  # (N, k, 3)
        scale_grad_sq = (scale_diff / neighbor_distances.unsqueeze(-1)) ** 2  # (N, k, 3)
        scale_smoothness = torch.sum(scale_grad_sq)  # Σᵢ ||∇sᵢ||²
        
        # Normalize by N*k to keep loss scale independent of N and k
        num_edges = N * k
        return self.lambda_s * (weight_smoothness + scale_smoothness) / num_edges


class TotalLoss(nn.Module):
    """
    Total Loss combining reconstruction and regularization terms.
    
    Mathematical Formulation
    ------------------------
    The total loss is a weighted combination of all loss components:
    
        L_total = L_mse + λ_s·L_sparse + λ_o·L_overlap + λ_sm·L_smooth
    
    Expanded form:
    
        L_total = (1/M)·Σₖ(f(pₖ) - vₖ)²           [Reconstruction]
                + λ_s·Σᵢ|wᵢ|                       [Sparsity]
                + λ_o·Σᵢ<ⱼ O(Gᵢ,Gⱼ)               [Overlap]
                + λ_sm·Σᵢ Σⱼ∈𝒩(i) ||θᵢ-θⱼ||²      [Smoothness]
    
    Hyperparameter Balancing
    ------------------------
    The regularization weights should be chosen such that:
    
    1. L_mse dominates initially (focus on fitting data)
    2. Regularization prevents overfitting/degeneracy
    3. Typical ranges:
        - λ_s ∈ [1e-5, 1e-2] : sparsity
        - λ_o ∈ [1e-5, 1e-2] : overlap
        - λ_sm ∈ [1e-5, 1e-2] : smoothness
    
    Training Dynamics
    -----------------
    Early training: L_mse >> regularization terms
        → Model focuses on reducing reconstruction error
        
    Late training: L_mse ≈ regularization terms (ideally)
        → Regularization refines solution quality
    
    If regularization dominates early:
        → Increase learning rate or decrease λ values
        
    If regularization has no effect:
        → Increase λ values or model may be underconstrained
    
    Loss Landscape Considerations
    -----------------------------
    - L_mse: Convex in weights, non-convex in positions/scales
    - L_sparse: Convex (L1 norm), creates sparse optima
    - L_overlap: Non-convex, can have many local minima
    - L_smooth: Convex (quadratic), acts as Laplacian smoother
    
    The combined loss is non-convex, requiring careful initialization
    and learning rate scheduling for good convergence.
    """
    
    def __init__(
        self,
        lambda_sparsity: float = 0.01,
        lambda_overlap: float = 0.01,
        lambda_smoothness: float = 0.01,
        use_sparsity: bool = True,
        use_overlap: bool = True,
        use_smoothness: bool = True
    ):
        """
        Initialize total loss with configurable regularization terms.
        
        Args:
            lambda_sparsity: Weight for L1 sparsity regularization (default: 0.01)
            lambda_overlap: Weight for overlap regularization (default: 0.01)
            lambda_smoothness: Weight for smoothness regularization (default: 0.01)
            use_sparsity: Enable sparsity regularization (default: True)
            use_overlap: Enable overlap regularization (default: True)
            use_smoothness: Enable smoothness regularization (default: True)
            
        Notes:
            Set λ=0 or use_X=False to disable specific regularization terms.
            Start with small λ values and increase if needed.
        """
        super().__init__()
        
        self.reconstruction_loss = ReconstructionLoss()
        
        self.use_sparsity = use_sparsity
        self.use_overlap = use_overlap
        self.use_smoothness = use_smoothness
        
        if use_sparsity:
            self.sparsity_loss = SparsityRegularization(lambda_sparsity)
        if use_overlap:
            self.overlap_loss = OverlapRegularization(lambda_overlap)
        if use_smoothness:
            self.smoothness_loss = SmoothnessRegularization(lambda_smoothness)
    
    def forward(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor,
        model=None
    ) -> dict:
        """
        Compute total loss with all enabled components.
        
        Args:
            predicted: Predicted voxel values of shape (M,) or (D,H,W)
            ground_truth: Ground truth voxel values, same shape as predicted
            model: GaussianVolumeModel instance (required for regularization)
            
        Returns:
            Dictionary containing:
                - 'mse': Reconstruction loss (always present)
                - 'sparsity': L1 sparsity loss (if enabled)
                - 'overlap': Overlap regularization (if enabled)
                - 'smoothness': Smoothness regularization (if enabled)
                - 'total': Sum of all enabled losses
                
        Example:
            >>> loss_fn = TotalLoss(lambda_sparsity=0.01)
            >>> losses = loss_fn(pred, gt, model)
            >>> losses['total'].backward()
        """
        losses = {}
        
        # Main reconstruction loss: L_mse = (1/M)·Σ(pred - gt)²
        losses['mse'] = self.reconstruction_loss(predicted, ground_truth)
        losses['total'] = losses['mse']
        
        if model is not None:
            # Sparsity regularization: L_sparse = λ·Σ|wᵢ|
            if self.use_sparsity:
                losses['sparsity'] = self.sparsity_loss(model.weights)
                losses['total'] = losses['total'] + losses['sparsity']
            
            # Overlap regularization: L_overlap = λ·Σᵢ<ⱼ O(Gᵢ,Gⱼ)
            if self.use_overlap:
                covariance = model.gaussians.get_covariance_matrices()
                losses['overlap'] = self.overlap_loss(model.positions, covariance)
                losses['total'] = losses['total'] + losses['overlap']
            
            # Smoothness regularization: L_smooth = λ·Σᵢ Σⱼ∈𝒩(i) ||θᵢ-θⱼ||²
            if self.use_smoothness:
                losses['smoothness'] = self.smoothness_loss(
                    model.positions, model.weights, model.log_scales
                )
                losses['total'] = losses['total'] + losses['smoothness']
        
        return losses


def compute_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Mathematical Definition
    -----------------------
    PSNR measures the ratio between the maximum possible signal power 
    and the power of corrupting noise (reconstruction error):
    
        PSNR = 10 · log₁₀(MAX² / MSE)
             = 20 · log₁₀(MAX / √MSE)
             = 20 · log₁₀(MAX) - 10 · log₁₀(MSE)
    
    where:
        - MAX : maximum possible pixel/voxel value
        - MSE : Mean Squared Error = (1/M)·Σ(pred - gt)²
    
    Interpretation
    --------------
    PSNR is expressed in decibels (dB). Higher is better.
    
    Typical ranges for image/volume reconstruction:
        - < 20 dB  : Poor quality, significant artifacts
        - 20-30 dB : Acceptable quality
        - 30-40 dB : Good quality
        - > 40 dB  : Excellent quality (often visually lossless)
    
    Relationship to SSIM
    --------------------
    PSNR measures pixel-wise error but doesn't capture perceptual quality.
    SSIM (Structural Similarity Index) is often used alongside PSNR:
    
        SSIM = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
    
    where l, c, s measure luminance, contrast, and structure similarity.
    
    For volumetric data (especially microscopy), PSNR is often preferred
    as structural assumptions of SSIM may not apply.
    
    Args:
        predicted: Predicted values (any shape, will be flattened)
        ground_truth: Ground truth values (same shape as predicted)
        
    Returns:
        PSNR value in decibels (dB). Returns inf if MSE = 0.
        
    Example:
        >>> psnr = compute_psnr(reconstructed_volume, original_volume)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    mse = F.mse_loss(predicted, ground_truth).item()
    if mse == 0:
        return float('inf')
    
    # Use max of ground truth as the peak signal value
    max_val = ground_truth.max().item()
    
    # PSNR = 20·log₁₀(MAX/√MSE)
    psnr = 20 * torch.log10(torch.tensor(max_val / (mse ** 0.5)))
    return psnr.item()


def compute_ssim(
    predicted: torch.Tensor, 
    ground_truth: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2
) -> float:
    """
    Compute Structural Similarity Index (SSIM) for 3D volumes.
    
    Mathematical Definition
    -----------------------
    SSIM compares local patterns of pixel intensities normalized for
    luminance and contrast:
    
        SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
    
    where:
        - μₓ, μᵧ : local means
        - σₓ², σᵧ² : local variances  
        - σₓᵧ : local covariance
        - C₁, C₂ : stability constants (avoid division by zero)
    
    The overall SSIM is averaged over all local windows.
    
    Components
    ----------
    SSIM can be decomposed into three components:
    
    1. Luminance: l(x,y) = (2μₓμᵧ + C₁)/(μₓ² + μᵧ² + C₁)
    2. Contrast:  c(x,y) = (2σₓσᵧ + C₂)/(σₓ² + σᵧ² + C₂)  
    3. Structure: s(x,y) = (σₓᵧ + C₃)/(σₓσᵧ + C₃)
    
    where C₃ = C₂/2. Full SSIM = l · c · s.
    
    Args:
        predicted: Predicted volume of shape (D, H, W)
        ground_truth: Ground truth volume of shape (D, H, W)
        window_size: Size of local window (default: 11)
        C1: Luminance stability constant
        C2: Contrast stability constant
        
    Returns:
        SSIM value in range [-1, 1]. Higher is better, 1 = identical.
        
    Note:
        This is a simplified implementation. For production use,
        consider pytorch-msssim or skimage.metrics.structural_similarity.
    """
    # Ensure 5D for 3D convolution: (B, C, D, H, W)
    if predicted.dim() == 3:
        predicted = predicted.unsqueeze(0).unsqueeze(0)
        ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        # 3D window = outer product of 1D windows
        window = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        return window.unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size).to(predicted.device)
    
    # Local means
    mu_x = F.conv3d(predicted, window, padding=window_size//2)
    mu_y = F.conv3d(ground_truth, window, padding=window_size//2)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    # Local variances and covariance
    sigma_x_sq = F.conv3d(predicted**2, window, padding=window_size//2) - mu_x_sq
    sigma_y_sq = F.conv3d(ground_truth**2, window, padding=window_size//2) - mu_y_sq
    sigma_xy = F.conv3d(predicted * ground_truth, window, padding=window_size//2) - mu_xy
    
    # SSIM formula
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-8)
    
    return ssim_map.mean().item()
