"""
Gaussian-Based Volume Data Representation Model

Implementation based on [21 Dec. 24] Algorithm from research proposal.

This module defines the Gaussian basis functions and the implicit neural
representation for volume data.
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianBasisFunctions(nn.Module):
    """
    Gaussian Basis Functions for Volume Data Representation.
    
    Each Gaussian is defined as:
        G_i(x, y, z; u_i, Σ_i) = exp{-0.5 * (x⃗ - u_i)^T * Σ_i^{-1} * (x⃗ - u_i)}
    
    Parameters:
        N: Number of Gaussian basis functions
        u_i: Positions (means) of Gaussians
        Σ_i: Covariance matrices (control size and orientation)
        w_i: Weights (refer to voxel values)
    """
    
    def __init__(
        self,
        num_gaussians: int,
        volume_shape: tuple,
        init_method: str = 'uniform',
        device: str = 'cuda'
    ):
        """
        Initialize N Gaussian basis functions.
        
        Args:
            num_gaussians: Number of Gaussian basis functions (N)
            volume_shape: Shape of the volume data (D, H, W)
            init_method: Initialization method ('uniform' or 'grid')
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.N = num_gaussians
        self.volume_shape = volume_shape
        self.device = device
        
        # Initialize positions (means) u_i
        # Set initial positions uniformly or based on voxel grid structure
        if init_method == 'uniform':
            # Random uniform initialization within volume bounds
            positions = torch.rand(num_gaussians, 3, device=device)
            # Scale to volume dimensions
            positions[:, 0] *= volume_shape[0]  # Depth
            positions[:, 1] *= volume_shape[1]  # Height
            positions[:, 2] *= volume_shape[2]  # Width
        elif init_method == 'grid':
            # Grid-based initialization
            n_per_dim = int(np.ceil(num_gaussians ** (1/3)))
            grid_points = []
            for d in np.linspace(0, volume_shape[0], n_per_dim):
                for h in np.linspace(0, volume_shape[1], n_per_dim):
                    for w in np.linspace(0, volume_shape[2], n_per_dim):
                        grid_points.append([d, h, w])
            positions = torch.tensor(grid_points[:num_gaussians], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        # Learnable parameter: u_i (centers/means of Gaussians)
        self.positions = nn.Parameter(positions)
        
        # Initialize covariance matrices Σ_i
        # Use log-scale factors for numerical stability
        # Σ = R * diag(s)^2 * R^T where s = exp(log_scales)
        # For simplicity, start with isotropic Gaussians (diagonal covariance)
        # log_scales controls size, initialized to reasonable values
        initial_scale = min(volume_shape) / (num_gaussians ** (1/3)) * 0.5
        log_scales = torch.full((num_gaussians, 3), np.log(initial_scale), device=device)
        self.log_scales = nn.Parameter(log_scales)
        
        # Rotation quaternions for orientation (w, x, y, z)
        # Initialize to identity rotation
        rotations = torch.zeros(num_gaussians, 4, device=device)
        rotations[:, 0] = 1.0  # w = 1, (x, y, z) = 0 for identity
        self.rotations = nn.Parameter(rotations)
        
        # Initialize weights w_i (refer to voxel values)
        # Initialize randomly or to small uniform values
        weights = torch.rand(num_gaussians, device=device) * 0.1
        self.weights = nn.Parameter(weights)
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute covariance matrices from scales and rotations.
        
        Σ_i = R_i * diag(s_i^2) * R_i^T
        
        Returns:
            Covariance matrices of shape (N, 3, 3)
        """
        # Get scales from log_scales
        scales = torch.exp(self.log_scales)  # (N, 3)
        
        # Normalize quaternions
        quats = self.rotations / (torch.norm(self.rotations, dim=1, keepdim=True) + 1e-8)
        
        # Convert quaternions to rotation matrices
        R = self._quaternion_to_rotation_matrix(quats)  # (N, 3, 3)
        
        # Construct diagonal scale matrix
        S = torch.diag_embed(scales ** 2)  # (N, 3, 3)
        
        # Covariance: Σ = R * S * R^T
        covariance = torch.bmm(torch.bmm(R, S), R.transpose(1, 2))  # (N, 3, 3)
        
        return covariance
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quaternions: Tensor of shape (N, 4) with (w, x, y, z)
            
        Returns:
            Rotation matrices of shape (N, 3, 3)
        """
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Rotation matrix from quaternion
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
        
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)
        
        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - x*w)
        
        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    def evaluate_gaussian(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all Gaussian basis functions at given points.
        
        G_i(x, y, z; u_i, Σ_i) = exp{-0.5 * (x⃗ - u_i)^T * Σ_i^{-1} * (x⃗ - u_i)}
        
        Args:
            points: Tensor of shape (M, 3) containing voxel coordinates (x, y, z)
            
        Returns:
            Gaussian values of shape (M, N) for each point and each Gaussian
        """
        M = points.shape[0]
        N = self.N
        
        # Get covariance matrices
        covariance = self.get_covariance_matrices()  # (N, 3, 3)
        
        # Compute inverse covariance matrices
        # Add small diagonal for numerical stability
        cov_inv = torch.linalg.inv(covariance + 1e-6 * torch.eye(3, device=self.device).unsqueeze(0))  # (N, 3, 3)
        
        # Compute difference (x - u_i) for all points and all Gaussians
        # points: (M, 3), positions: (N, 3)
        # diff: (M, N, 3)
        diff = points.unsqueeze(1) - self.positions.unsqueeze(0)  # (M, N, 3)
        
        # Compute quadratic form: (x - u)^T * Σ^{-1} * (x - u)
        # diff: (M, N, 3), cov_inv: (N, 3, 3)
        # quad_form: (M, N)
        quad_form = torch.einsum('mni,nij,mnj->mn', diff, cov_inv, diff)
        
        # Gaussian: exp(-0.5 * quad_form)
        gaussian_values = torch.exp(-0.5 * quad_form)  # (M, N)
        
        return gaussian_values
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Construct the implicit function f(x, y, z).
        
        f(x, y, z) = Σ_{i=1}^{N} w_i * G_i(x, y, z; u_i, Σ_i)
        
        Args:
            points: Tensor of shape (M, 3) containing voxel coordinates
            
        Returns:
            Function values of shape (M,)
        """
        # Evaluate all Gaussians at all points
        gaussian_values = self.evaluate_gaussian(points)  # (M, N)
        
        # Weighted sum: f(x) = Σ w_i * G_i(x)
        # weights: (N,), gaussian_values: (M, N)
        output = torch.matmul(gaussian_values, self.weights)  # (M,)
        
        return output
    
    def get_parameters_dict(self) -> dict:
        """
        Get all learnable parameters as a dictionary.
        
        Returns:
            Dictionary containing positions, log_scales, rotations, and weights
        """
        return {
            'positions': self.positions.data.clone(),
            'log_scales': self.log_scales.data.clone(),
            'rotations': self.rotations.data.clone(),
            'weights': self.weights.data.clone(),
            'num_gaussians': self.N,
            'volume_shape': self.volume_shape
        }
    
    def load_parameters_dict(self, params_dict: dict):
        """
        Load parameters from a dictionary.
        
        Args:
            params_dict: Dictionary containing saved parameters
        """
        self.positions.data = params_dict['positions']
        self.log_scales.data = params_dict['log_scales']
        self.rotations.data = params_dict['rotations']
        self.weights.data = params_dict['weights']


class GaussianVolumeModel(nn.Module):
    """
    Complete Gaussian-Based Volume Data Representation Model.
    
    This class wraps GaussianBasisFunctions and provides a complete
    interface for training and inference.
    """
    
    def __init__(
        self,
        num_gaussians: int,
        volume_shape: tuple,
        init_method: str = 'uniform',
        device: str = 'cuda'
    ):
        """
        Initialize the Gaussian Volume Model.
        
        Args:
            num_gaussians: Number of Gaussian basis functions (N)
            volume_shape: Shape of the volume data (D, H, W)
            init_method: Initialization method ('uniform' or 'grid')
            device: Device to use
        """
        super().__init__()
        
        self.gaussians = GaussianBasisFunctions(
            num_gaussians=num_gaussians,
            volume_shape=volume_shape,
            init_method=init_method,
            device=device
        )
        
        self.device = device
        self.volume_shape = volume_shape
        self.num_gaussians = num_gaussians
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the implicit function at given points.
        
        Args:
            points: Voxel coordinates of shape (M, 3)
            
        Returns:
            Predicted voxel values of shape (M,)
        """
        return self.gaussians(points)
    
    def reconstruct_volume(self, batch_size: int = 10000) -> torch.Tensor:
        """
        Reconstruct the entire volume from Gaussian representation.
        
        Args:
            batch_size: Batch size for evaluation to manage memory
            
        Returns:
            Reconstructed volume tensor
        """
        D, H, W = self.volume_shape
        
        # Create coordinate grid
        d_coords = torch.arange(D, device=self.device, dtype=torch.float32)
        h_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        w_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        
        # Create meshgrid
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        points = torch.stack([grid_d.flatten(), grid_h.flatten(), grid_w.flatten()], dim=1)
        
        # Evaluate in batches
        output = torch.zeros(points.shape[0], device=self.device)
        
        with torch.no_grad():
            for i in range(0, points.shape[0], batch_size):
                batch_points = points[i:i+batch_size]
                output[i:i+batch_size] = self.forward(batch_points)
        
        # Reshape to volume
        volume = output.reshape(D, H, W)
        
        return volume
    
    @property
    def positions(self):
        return self.gaussians.positions
    
    @property
    def weights(self):
        return self.gaussians.weights
    
    @property
    def log_scales(self):
        return self.gaussians.log_scales
    
    @property
    def rotations(self):
        return self.gaussians.rotations
