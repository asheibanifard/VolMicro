"""
Trainer for Gaussian-Based Volume Data Representation

Implementation based on [21 Dec. 24] Algorithm from research proposal.

This module implements the iterative training loop:
1. Sample voxel data
2. Forward pass (evaluate Gaussians)
3. Compute loss
4. Backpropagation
5. Update parameters

Gradient-based optimization using Adam or SGD.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Dict, Tuple, List

from .gaussian_model import GaussianVolumeModel
from .losses import TotalLoss, compute_psnr


class VolumeDataSampler:
    """
    Sample voxel data from volume.
    
    Step 5.1 from algorithm:
    - Randomly sample a batch of voxel coordinates (x_k, y_k, z_k)
    - Retrieve their corresponding ground truth values v_k
    """
    
    def __init__(
        self,
        volume: torch.Tensor,
        device: str = 'cuda'
    ):
        """
        Initialize sampler with volume data.
        
        Args:
            volume: Volume tensor of shape (D, H, W)
            device: Device to use
        """
        self.volume = volume.to(device)
        self.device = device
        self.D, self.H, self.W = self.volume.shape
        
        # Create coordinate grid
        d_coords = torch.arange(self.D, device=device, dtype=torch.float32)
        h_coords = torch.arange(self.H, device=device, dtype=torch.float32)
        w_coords = torch.arange(self.W, device=device, dtype=torch.float32)
        
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Flatten to get all coordinates
        self.all_coords = torch.stack([
            grid_d.flatten(),
            grid_h.flatten(),
            grid_w.flatten()
        ], dim=1)  # (D*H*W, 3)
        
        self.all_values = self.volume.flatten()  # (D*H*W,) - use self.volume which is on device
        self.total_voxels = self.D * self.H * self.W
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of voxel coordinates and values.
        
        Args:
            batch_size: Number of voxels to sample
            
        Returns:
            Tuple of (coordinates, values), shapes (batch_size, 3) and (batch_size,)
        """
        # Random indices
        indices = torch.randint(0, self.total_voxels, (batch_size,), device=self.device)
        
        # Get coordinates and values
        coords = self.all_coords[indices]
        values = self.all_values[indices]
        
        return coords, values
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all voxel coordinates and values.
        
        Returns:
            Tuple of (all_coords, all_values)
        """
        return self.all_coords, self.all_values
    
    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader for the volume data.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for (coordinates, values) pairs
        """
        dataset = TensorDataset(self.all_coords, self.all_values)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GaussianVolumeTrainer:
    """
    Trainer for Gaussian Volume Model.
    
    Implements Steps 4-5 from the algorithm:
    - Gradient-based optimization (Adam/SGD)
    - Backpropagation through Gaussian functions
    - Iterative training with batched voxel sampling
    """
    
    def __init__(
        self,
        model: GaussianVolumeModel,
        volume: torch.Tensor,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam',
        lambda_sparsity: float = 0.01,
        lambda_overlap: float = 0.01,
        lambda_smoothness: float = 0.01,
        use_sparsity: bool = True,
        use_overlap: bool = True,
        use_smoothness: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: GaussianVolumeModel to train
            volume: Ground truth volume tensor of shape (D, H, W)
            learning_rate: Learning rate (ρ in the algorithm)
            optimizer_type: 'adam' or 'sgd'
            lambda_sparsity: Weight for sparsity regularization (λ_w)
            lambda_overlap: Weight for overlap regularization (λ_o)
            lambda_smoothness: Weight for smoothness regularization (λ_s)
            use_sparsity: Whether to use sparsity regularization
            use_overlap: Whether to use overlap regularization
            use_smoothness: Whether to use smoothness regularization
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Data sampler
        self.sampler = VolumeDataSampler(volume, device)
        self.volume = volume.to(device)
        
        # Optimizer (Step 4.1: Use Adam or SGD)
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Loss function
        self.loss_fn = TotalLoss(
            lambda_sparsity=lambda_sparsity,
            lambda_overlap=lambda_overlap,
            lambda_smoothness=lambda_smoothness,
            use_sparsity=use_sparsity,
            use_overlap=use_overlap,
            use_smoothness=use_smoothness
        )
        
        # Training history
        self.history = {
            'total_loss': [],
            'mse_loss': [],
            'sparsity_loss': [],
            'overlap_loss': [],
            'smoothness_loss': [],
            'psnr': []
        }
    
    def train_step(self, coords: torch.Tensor, values: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Steps 5.2-5.5:
        - Forward pass: Compute f(x_k)
        - Compute loss L
        - Backpropagate to compute gradients
        - Update parameters with optimizer
        
        Args:
            coords: Voxel coordinates of shape (batch_size, 3)
            values: Ground truth voxel values of shape (batch_size,)
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Step 5.2: Forward Pass
        # For each voxel coordinate: f(x⃗_k) = Σ w_i * G_i(x⃗_k)
        predicted = self.model(coords)
        
        # Step 5.3: Compute Loss
        losses = self.loss_fn(predicted, values, self.model)
        
        # Step 5.4: Backpropagation
        # Compute gradients for: w_i, u_i, Σ_i
        losses['total'].backward()
        
        # Step 5.5: Update Parameters
        # w_i ← w_i - ρ * ∂L/∂w_i
        # Σ_i ← Σ_i - ρ * ∂L/∂Σ_i
        # u_i ← u_i - ρ * ∂L/∂u_i
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def train_epoch(
        self,
        batch_size: int = 4096,
        use_dataloader: bool = False,
        batches_per_epoch: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            batch_size: Batch size for sampling
            use_dataloader: Whether to use DataLoader (full coverage) or random sampling
            batches_per_epoch: Number of batches when using random sampling (faster)
            
        Returns:
            Dictionary of average loss values for the epoch
        """
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'sparsity': 0.0,
            'overlap': 0.0,
            'smoothness': 0.0
        }
        num_batches = 0
        
        if use_dataloader:
            dataloader = self.sampler.get_dataloader(batch_size, shuffle=True)
            for coords, values in dataloader:
                coords = coords.to(self.device)
                values = values.to(self.device)
                losses = self.train_step(coords, values)
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
                num_batches += 1
        else:
            # Random sampling approach (FASTER - doesn't iterate all voxels)
            for _ in range(batches_per_epoch):
                coords, values = self.sampler.sample_batch(batch_size)
                losses = self.train_step(coords, values)
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 4096,
        eval_interval: int = 10,
        save_interval: int = 50,
        save_dir: str = 'checkpoints',
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop (Step 5: Iterative Training).
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for sampling
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        pbar = tqdm(range(num_epochs), disable=not verbose)
        
        for epoch in pbar:
            # Train for one epoch
            epoch_losses = self.train_epoch(batch_size)
            
            # Record history
            self.history['total_loss'].append(epoch_losses['total'])
            self.history['mse_loss'].append(epoch_losses['mse'])
            if 'sparsity' in epoch_losses:
                self.history['sparsity_loss'].append(epoch_losses['sparsity'])
            if 'overlap' in epoch_losses:
                self.history['overlap_loss'].append(epoch_losses['overlap'])
            if 'smoothness' in epoch_losses:
                self.history['smoothness_loss'].append(epoch_losses['smoothness'])
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0:
                psnr = self.evaluate()
                self.history['psnr'].append(psnr)
                
                pbar.set_postfix({
                    'loss': f"{epoch_losses['total']:.4e}",
                    'mse': f"{epoch_losses['mse']:.4e}",
                    'psnr': f"{psnr:.2f}dB"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{epoch_losses['total']:.4e}",
                    'mse': f"{epoch_losses['mse']:.4e}"
                })
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'),
                    epoch + 1
                )
        
        # Final save
        self.save_checkpoint(
            os.path.join(save_dir, 'model_final.pth'),
            num_epochs
        )
        
        return self.history
    
    def evaluate(self, batch_size: int = 10000) -> float:
        """
        Evaluate model on entire volume.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            PSNR value
        """
        self.model.eval()
        
        coords, values = self.sampler.get_all_data()
        
        predictions = []
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i:i+batch_size]
                pred = self.model(batch_coords)
                predictions.append(pred)
        
        predictions = torch.cat(predictions)
        psnr = compute_psnr(predictions, values)
        
        self.model.train()
        return psnr
    
    def save_checkpoint(self, path: str, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'num_gaussians': self.model.num_gaussians,
            'volume_shape': self.model.volume_shape
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


def create_trainer(
    volume: torch.Tensor,
    num_gaussians: int,
    learning_rate: float = 0.01,
    optimizer_type: str = 'adam',
    lambda_sparsity: float = 0.01,
    lambda_overlap: float = 0.01,
    lambda_smoothness: float = 0.01,
    init_method: str = 'uniform',
    device: str = 'cuda'
) -> Tuple[GaussianVolumeModel, GaussianVolumeTrainer]:
    """
    Helper function to create model and trainer.
    
    Args:
        volume: Ground truth volume tensor
        num_gaussians: Number of Gaussian basis functions
        learning_rate: Learning rate
        optimizer_type: Optimizer type ('adam' or 'sgd')
        lambda_sparsity: Sparsity regularization weight
        lambda_overlap: Overlap regularization weight
        lambda_smoothness: Smoothness regularization weight
        init_method: Gaussian initialization method
        device: Device to use
        
    Returns:
        Tuple of (model, trainer)
    """
    volume_shape = volume.shape
    
    model = GaussianVolumeModel(
        num_gaussians=num_gaussians,
        volume_shape=volume_shape,
        init_method=init_method,
        device=device
    )
    
    trainer = GaussianVolumeTrainer(
        model=model,
        volume=volume,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        lambda_sparsity=lambda_sparsity,
        lambda_overlap=lambda_overlap,
        lambda_smoothness=lambda_smoothness,
        device=device
    )
    
    return model, trainer
