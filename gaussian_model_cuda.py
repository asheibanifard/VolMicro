"""
CUDA-Accelerated Gaussian Basis Volume Model (Medical Volume Fitting)

Represents a scalar 3D volume as a sum of anisotropic 3D Gaussians:

    f(x) = Σ_i  a_i * exp( -0.5 * (x - μ_i)^T Σ_i^{-1} (x - μ_i) )

Key design choices (for correctness + stability):
- Positions are constrained to (0,1) using sigmoid on a raw parameter.
- Scales are constrained to >0 using softplus (no abs kink).
- Intensities are consistently activated (choose one):
    * sigmoid  -> output contributions in (0,1) (for volumes normalised to [0,1])
    * softplus -> nonnegative unbounded
    * linear   -> signed (requires stronger regularisation)
- Opacities are optional and kept in sync during densify/prune (for later 3DGS export).
- Grid uses voxel-centre coordinates, not boundaries.

Sparse Evaluation (KNN):
- For large N, use forward_knn() to evaluate only K nearest Gaussians per query point
- Uses simple_knn module (from 3DGS) for CUDA-accelerated mean distance computation
- Falls back to PyTorch cdist for KNN queries when simple_knn unavailable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gs_utils.Compute_intensity import compute_intensity

# Try to import simple_knn for CUDA-accelerated KNN distance computation
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'temp_gs', 'submodules', 'simple-knn'))
    from simple_knn._C import distCUDA2
    SIMPLE_KNN_AVAILABLE = True
except ImportError:
    SIMPLE_KNN_AVAILABLE = False
    print("Warning: simple_knn not available, using PyTorch fallback for KNN operations")

# Try to import FAISS for GPU-accelerated KNN
try:
    import faiss
    import faiss.contrib.torch_utils  # Enable direct torch tensor support
    FAISS_AVAILABLE = True
    # Check for GPU support
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    if FAISS_GPU_AVAILABLE:
        print(f"FAISS GPU available with {faiss.get_num_gpus()} GPU(s)")
    else:
        print("FAISS available (CPU only)")
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    print("Warning: FAISS not available, using PyTorch fallback for KNN (slower)")


class CUDAGaussianModel(nn.Module):
    def __init__(
        self,
        num_gaussians: int,
        volume_shape: tuple,          # (D, H, W)
        init_method: str = "uniform", # "uniform", "grid", or "swc"
        device: str = "cuda",
        intensity_activation: str = "sigmoid",  # "sigmoid" | "softplus" | "linear"
        init_opacity: float = 0.1,    # used only for 3DGS-style export
        swc_path: str = None,         # path to SWC file for skeleton initialization
        swc_densify: bool = True,     # densify skeleton points
    ):
        super().__init__()
        if device != "cuda":
            raise ValueError("CUDAGaussianModel requires CUDA device")

        self.volume_shape = tuple(volume_shape)
        self.device = device
        self.D, self.H, self.W = self.volume_shape

        if intensity_activation not in ("sigmoid", "softplus", "linear"):
            raise ValueError("intensity_activation must be 'sigmoid', 'softplus', or 'linear'")
        self.intensity_activation = intensity_activation

        # ----- initialise positions in [0,1], but store as unconstrained raw params -----
        if init_method == "uniform":
            self.N = int(num_gaussians)
            positions_01 = torch.rand(self.N, 3, device=device)
            init_scales = None
        elif init_method == "grid":
            self.N = int(num_gaussians)
            n_per_dim = int(np.ceil(self.N ** (1/3)))
            grid_points = []
            for d in np.linspace(0, 1, n_per_dim):
                for h in np.linspace(0, 1, n_per_dim):
                    for w in np.linspace(0, 1, n_per_dim):
                        grid_points.append([d, h, w])
            positions_01 = torch.tensor(grid_points[: self.N], dtype=torch.float32, device=device)
            init_scales = None
        elif init_method == "swc":
            if swc_path is None:
                raise ValueError("swc_path required for init_method='swc'")
            positions_01, init_scales = self._init_from_swc(
                swc_path, num_gaussians, swc_densify
            )
            self.N = positions_01.shape[0]
            print(f"  Initialized {self.N} Gaussians from SWC skeleton")
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # inverse-sigmoid to initialise raw positions so sigmoid(raw)=positions_01
        eps = 1e-6
        positions_01 = positions_01.clamp(eps, 1 - eps)
        positions_raw = torch.log(positions_01) - torch.log(1 - positions_01)
        self.positions_raw = nn.Parameter(positions_raw)  # (N,3)

        # ----- scales: store raw and activate by softplus -----
        if init_scales is not None:
            # Use SWC radii as initial scales
            scales_raw = torch.log(torch.exp(init_scales) - 1.0 + 1e-6)  # inverse softplus
        else:
            # initial scale heuristic: small but not tiny
            init_scale = float(0.10 / (self.N ** (1/3)))
            scales_raw = torch.full((self.N, 3), np.log(np.exp(init_scale) - 1.0), device=device)
        self.scales_raw = nn.Parameter(scales_raw)  # (N,3)

        # ----- rotations (quaternions) -----
        rotations = torch.zeros(self.N, 4, device=device, dtype=torch.float32)
        rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(rotations)  # (N,4)

        # ----- intensities (raw); activation applied consistently everywhere -----
        intensities_raw = torch.randn(self.N, device=device) * 0.1
        self.intensities_raw = nn.Parameter(intensities_raw)  # (N,)

        # ----- opacities for 3DGS export (optional, but kept consistent) -----
        # raw_opacity init = inverse sigmoid(init_opacity)
        init_opacity = float(np.clip(init_opacity, 1e-4, 1 - 1e-4))
        raw_op = np.log(init_opacity) - np.log(1 - init_opacity)
        self.raw_opacities = nn.Parameter(torch.full((self.N,), raw_op, device=device))

        # FAISS index cache (rebuilt periodically, not every iteration)
        self._faiss_index = None
        self._faiss_index_positions = None  # Positions when index was built
        
        self._setup_grid()

    def _init_from_swc(self, swc_path: str, max_gaussians: int, densify: bool
                       ) -> tuple:
        """
        Initialize Gaussian positions from SWC skeleton file.
        
        Args:
            swc_path: Path to SWC file
            max_gaussians: Maximum number of Gaussians to create (0 = use all SWC points)
            densify: Whether to interpolate points along skeleton segments
            
        Returns:
            positions: (N, 3) tensor in [0, 1]
            scales: (N, 3) tensor of initial scales based on radii
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from swc_utils import parse_swc_file, swc_to_arrays, densify_skeleton, normalize_positions
        
        print(f"  Loading SWC skeleton from: {swc_path}")
        nodes = parse_swc_file(swc_path)
        positions, radii, parent_ids = swc_to_arrays(nodes)
        print(f"  SWC nodes: {len(nodes)}")
        
        # Use all points if max_gaussians is 0 or less than node count
        use_all = (max_gaussians <= 0) or (max_gaussians <= len(positions))
        
        if densify and not use_all and len(positions) < max_gaussians:
            # Densify skeleton to get more points
            positions, radii = densify_skeleton(
                positions, radii, parent_ids,
                points_per_unit=max(1.0, max_gaussians / len(positions) / 10),
                min_points_per_segment=3
            )
            print(f"  After densification: {len(positions)} points")
        
        # Normalize to [0, 1] with volume shape consideration
        # SWC coords are typically in voxel space matching the volume
        D, H, W = self.volume_shape
        max_dim = max(D, H, W)
        
        # Normalize positions to [0, 1]
        positions_norm = normalize_positions(positions, margin=0.02)
        
        # Subsample if we have too many points and max_gaussians > 0
        if max_gaussians > 0 and len(positions_norm) > max_gaussians:
            indices = np.random.choice(len(positions_norm), max_gaussians, replace=False)
            positions_norm = positions_norm[indices]
            radii = radii[indices]
            print(f"  Subsampled to {max_gaussians} Gaussians")
        
        # Convert radii to scales (normalized)
        # Radii in SWC are in voxel units, normalize to [0, 1] space
        radii_norm = radii / max_dim
        radii_norm = np.clip(radii_norm, 0.005, 0.1)  # reasonable scale range
        
        # Convert to tensors
        positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=self.device)
        scales_tensor = torch.tensor(
            np.stack([radii_norm, radii_norm, radii_norm], axis=1),
            dtype=torch.float32, device=self.device
        )
        
        return positions_tensor, scales_tensor

    # -------------------- activated parameters --------------------
    @property
    def positions(self) -> torch.Tensor:
        # (N,3) in (0,1)
        return torch.sigmoid(self.positions_raw)

    @property
    def scales(self) -> torch.Tensor:
        # (N,3) > 0
        return F.softplus(self.scales_raw) + 1e-6

    @property
    def opacities(self) -> torch.Tensor:
        # (N,) in (0,1)
        return torch.sigmoid(self.raw_opacities)

    def intensities(self) -> torch.Tensor:
        # Consistent activation for CUDA + export
        if self.intensity_activation == "sigmoid":
            return torch.sigmoid(self.intensities_raw)
        if self.intensity_activation == "softplus":
            return F.softplus(self.intensities_raw)
        return self.intensities_raw  # linear/signed

    # -------------------- grid --------------------
    def _setup_grid(self) -> None:
        """Voxel-centre grid in [0,1]."""
        d = (torch.arange(self.D, device=self.device, dtype=torch.float32) + 0.5) / self.D
        h = (torch.arange(self.H, device=self.device, dtype=torch.float32) + 0.5) / self.H
        w = (torch.arange(self.W, device=self.device, dtype=torch.float32) + 0.5) / self.W

        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")
        grid = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # (D,H,W,3)
        self.register_buffer("grid_points", grid.reshape(1, self.D, self.H, self.W, 3).contiguous())

    # -------------------- rotations & covariances --------------------
    @staticmethod
    def _quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """(N,4) -> (N,3,3) with normalised quaternions."""
        q = quaternions / (torch.norm(quaternions, dim=1, keepdim=True) + 1e-8)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device, dtype=quaternions.dtype)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def get_inv_covariances(self) -> torch.Tensor:
        """
        Σ_i = R_i * diag(s_i^2) * R_i^T
        Σ_i^{-1} = R_i * diag(1/s_i^2) * R_i^T
        Returns (N,9).
        """
        s = self.scales  # (N,3) > 0
        R = self._quaternion_to_rotation_matrix(self.rotations)  # (N,3,3)
        inv_S = torch.diag_embed(1.0 / (s * s))  # (N,3,3)
        inv_cov = torch.bmm(torch.bmm(R, inv_S), R.transpose(1, 2))
        return inv_cov.reshape(self.N, 9).contiguous()

    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute covariance matrices: Σ_i = R_i * diag(s_i^2) * R_i^T
        Returns (N, 3, 3).
        """
        s = self.scales  # (N,3) > 0
        R = self._quaternion_to_rotation_matrix(self.rotations)  # (N,3,3)
        S = torch.diag_embed(s * s)  # (N,3,3) - variance is scale squared
        cov = torch.bmm(torch.bmm(R, S), R.transpose(1, 2))
        return cov

    def log_scales(self) -> torch.Tensor:
        """
        Return log of scales for smoothness regularization.
        Note: This model uses softplus(scales_raw) not exp(log_scales),
        so we compute log(scales) from activated scales.
        Returns (N, 3).
        """
        return torch.log(self.scales + 1e-8)

    # -------------------- export params (for later rasterisation) --------------------
    def get_3dgs_params(self) -> dict:
        """
        Export parameters in a 3DGS-like dict.
        Note: for scalar medical volumes, intensities are typically (N,1).
        """
        return {
            "positions": self.positions,                                 # (N,3) in (0,1)
            "scales": self.scales,                                       # (N,3) > 0
            "rotations": F.normalize(self.rotations, dim=1),              # (N,4)
            "opacities": self.opacities.unsqueeze(-1),                   # (N,1)
            "intensities": self.intensities().unsqueeze(-1),             # (N,1)
        }

    def get_parameters_dict(self) -> dict:
        """
        Get all model parameters for checkpointing/saving.
        Returns raw (unconstrained) parameters for exact restoration.
        """
        return {
            "num_gaussians": self.N,
            "volume_shape": self.volume_shape,
            "intensity_activation": self.intensity_activation,
            "positions_raw": self.positions_raw.data,
            "scales_raw": self.scales_raw.data,
            "rotations": self.rotations.data,
            "intensities_raw": self.intensities_raw.data,
            "raw_opacities": self.raw_opacities.data,
        }

    # -------------------- CUDA evaluation --------------------
    def forward(self) -> torch.Tensor:
        """
        Full volume reconstruction: returns (D,H,W).
        """
        centers = self.positions.contiguous()                # (N,3)
        grid_points_flat = self.grid_points.reshape(-1, 3).contiguous()  # (D*H*W,3)
        amps = self.intensities().contiguous()               # (N,)
        inv_cov = self.get_inv_covariances()                 # (N,9)
        scalings = self.scales.contiguous()                  # (N,3)

        out = torch.zeros(1, self.D, self.H, self.W, device=self.device, dtype=torch.float32)
        out = compute_intensity(centers, grid_points_flat, amps, inv_cov, scalings, out)
        return out.squeeze(0)

    def forward_sampled(self, sample_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate only at provided points (M,3) in (0,1).
        NOTE: This assumes your CUDA kernel treats grid_points_flat length as number of points.
        If your kernel assumes (D,H,W) indexing, this must be rewritten to a dedicated kernel.
        """
        if sample_points.ndim != 2 or sample_points.shape[1] != 3:
            raise ValueError("sample_points must be (M,3)")
        centers = self.positions.contiguous()
        pts = sample_points.contiguous()
        amps = self.intensities().contiguous()
        inv_cov = self.get_inv_covariances()
        scalings = self.scales.contiguous()

        M = pts.shape[0]
        out = torch.zeros(1, M, 1, 1, device=self.device, dtype=torch.float32)
        out = compute_intensity(centers, pts, amps, inv_cov, scalings, out)
        return out.flatten()

    # -------------------- KNN-based sparse evaluation --------------------
    def get_knn_mean_distances(self) -> torch.Tensor:
        """
        Compute mean distance to 3 nearest neighbors for each Gaussian.
        Uses simple_knn CUDA module if available, else PyTorch fallback.
        
        Returns:
            (N,) tensor of mean distances to 3-NN for each Gaussian
        """
        positions = self.positions  # (N, 3) in [0,1]
        
        if SIMPLE_KNN_AVAILABLE:
            # Use CUDA-accelerated KNN from 3DGS
            mean_dists = distCUDA2(positions.contiguous())
            return torch.sqrt(mean_dists.clamp(min=1e-8))  # distCUDA2 returns squared distances
        else:
            # PyTorch fallback: compute pairwise distances
            dists = torch.cdist(positions, positions)  # (N, N)
            # Set diagonal to inf to exclude self
            dists.fill_diagonal_(float('inf'))
            # Get 3 nearest neighbors
            knn_dists, _ = torch.topk(dists, k=min(3, self.N - 1), largest=False, dim=1)
            return knn_dists.mean(dim=1)  # (N,)

    def _build_faiss_index(self, positions: torch.Tensor):
        """
        Build FAISS index for fast KNN queries.
        Uses GPU if available, otherwise CPU.
        
        Args:
            positions: (N, 3) Gaussian positions
        """
        if not FAISS_AVAILABLE:
            return None
        
        N, d = positions.shape
        positions_np = positions.detach().cpu().numpy().astype(np.float32)
        
        # Use IVF only if we have enough points (FAISS recommends ~39 points per cluster minimum)
        # For N points with nlist clusters, we need N >= 39 * nlist
        use_ivf = N > 50000  # Only use IVF for large N
        
        if FAISS_GPU_AVAILABLE:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)  # 128MB temp memory
            
            if use_ivf:
                # IVF index: approximate but O(sqrt(N)) search
                # nlist should satisfy: N >= 39 * nlist, so nlist <= N / 39
                nlist = min(int(np.sqrt(N)), N // 50)  # Conservative cluster count
                nlist = max(nlist, 1)  # At least 1 cluster
                quantizer = faiss.IndexFlatL2(d)
                index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index_cpu.train(positions_np)
                index_cpu.add(positions_np)
                index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                index.nprobe = min(max(nlist // 4, 1), 32)  # search this many clusters
            else:
                # Flat index - exact but still very fast on GPU
                index_cpu = faiss.IndexFlatL2(d)
                index_cpu.add(positions_np)
                index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            # CPU index
            if use_ivf:
                nlist = min(int(np.sqrt(N)), N // 50)
                nlist = max(nlist, 1)
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                index.train(positions_np)
                index.add(positions_np)
                index.nprobe = min(max(nlist // 4, 1), 32)
            else:
                index = faiss.IndexFlatL2(d)
                index.add(positions_np)
        
        return index
    
    def get_knn_indices(self, query_points: torch.Tensor, k: int = 16, batch_size: int = 100000) -> torch.Tensor:
        """
        Find K nearest Gaussian centers for each query point.
        Uses FAISS GPU if available for ~10-100x speedup.
        
        Args:
            query_points: (M, 3) query positions in [0,1]
            k: Number of nearest neighbors
            batch_size: Process this many query points at a time (only for fallback)
            
        Returns:
            (M, K) indices of nearest Gaussians for each query point
        """
        positions = self.positions  # (N, 3)
        k = min(k, self.N)
        M = query_points.shape[0]
        
        # Use FAISS only for very large problems where rebuild is rare
        # For training with changing positions, PyTorch cdist on GPU is faster
        # because it avoids CPU<->GPU data transfer
        use_faiss = FAISS_AVAILABLE and (M * self.N > 5e8)  # Only for huge problems
        
        if use_faiss:
            # Check if we need to rebuild the index (positions changed significantly or N changed)
            need_rebuild = (
                self._faiss_index is None or
                self._faiss_index_positions is None or
                self._faiss_index_positions.shape[0] != self.N
            )
            
            if need_rebuild:
                self._faiss_index = self._build_faiss_index(positions)
                self._faiss_index_positions = positions.detach().clone()
            
            if self._faiss_index is not None:
                query_np = query_points.detach().cpu().numpy().astype(np.float32)
                _, knn_idx = self._faiss_index.search(query_np, k)
                return torch.from_numpy(knn_idx).long().to(query_points.device)
        
        # Fallback to PyTorch (for small problems or if FAISS unavailable)
        if M <= batch_size:
            # Small enough to do in one shot
            dists = torch.cdist(query_points, positions)
            _, knn_idx = torch.topk(dists, k=k, largest=False, dim=1)
            return knn_idx
        
        # Batch processing for large query sets
        all_indices = []
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch_query = query_points[start:end]
            dists = torch.cdist(batch_query, positions)
            _, knn_idx = torch.topk(dists, k=k, largest=False, dim=1)
            all_indices.append(knn_idx)
        
        return torch.cat(all_indices, dim=0)

    def forward_knn(self, query_points: torch.Tensor, k: int = 16, 
                    sigma_cutoff: float = 3.0, batch_size: int = 10000) -> torch.Tensor:
        """
        Sparse Gaussian evaluation using K-nearest neighbors.
        Uses batching to handle large query sets without OOM.
        
        For each query point, only evaluates the K nearest Gaussians,
        providing O(M*K) complexity instead of O(M*N).
        
        Args:
            query_points: (M, 3) query positions in [0,1]
            k: Number of nearest Gaussians to consider per query point
            sigma_cutoff: Additional cutoff - zero out contributions beyond this many σ
            batch_size: Process this many points at a time
            
        Returns:
            (M,) predicted intensity values
        """
        M = query_points.shape[0]
        k = min(k, self.N)
        
        if M <= batch_size:
            return self._forward_knn_batch(query_points, k, sigma_cutoff)
        
        # Batch processing for large query sets
        outputs = []
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch_query = query_points[start:end]
            batch_out = self._forward_knn_batch(batch_query, k, sigma_cutoff)
            outputs.append(batch_out)
        
        return torch.cat(outputs, dim=0)
    
    def _forward_knn_batch(self, query_points: torch.Tensor, k: int, 
                           sigma_cutoff: float) -> torch.Tensor:
        """Process a single batch of query points for KNN evaluation."""
        # Get KNN indices: (M, K)
        knn_idx = self.get_knn_indices(query_points, k)
        
        # Gather Gaussian parameters for KNN
        positions = self.positions  # (N, 3)
        amps = self.intensities()  # (N,)
        inv_cov = self.get_inv_covariances().reshape(self.N, 3, 3)  # (N, 3, 3)
        
        # Gather for each query's neighbors: (M, K, ...)
        knn_positions = positions[knn_idx]  # (M, K, 3)
        knn_amps = amps[knn_idx]  # (M, K)
        knn_inv_cov = inv_cov[knn_idx]  # (M, K, 3, 3)
        
        # Compute displacement: (M, K, 3)
        query_expanded = query_points.unsqueeze(1)  # (M, 1, 3)
        diff = query_expanded - knn_positions  # (M, K, 3)
        
        # Compute Mahalanobis distance squared: d^T * Σ^{-1} * d
        diff_expanded = diff.unsqueeze(-1)  # (M, K, 3, 1)
        mahal_sq = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), knn_inv_cov),  # (M, K, 1, 3)
            diff_expanded  # (M, K, 3, 1)
        ).squeeze(-1).squeeze(-1)  # (M, K)
        
        # Gaussian evaluation: exp(-0.5 * mahal_sq)
        gaussian_vals = torch.exp(-0.5 * mahal_sq)  # (M, K)
        
        # Apply sigma cutoff
        if sigma_cutoff is not None and sigma_cutoff > 0:
            cutoff_mask = mahal_sq < (sigma_cutoff ** 2)
            gaussian_vals = gaussian_vals * cutoff_mask.float()
        
        # Weighted sum: Σ a_i * G_i(x)
        return (knn_amps * gaussian_vals).sum(dim=1)  # (M,)

    def forward_knn_volume(self, k: int = 16, sigma_cutoff: float = 3.0,
                           batch_size: int = 100000) -> torch.Tensor:
        """
        Reconstruct full volume using KNN-sparse evaluation.
        
        For large N, this is more efficient than full forward() as it
        evaluates only K Gaussians per voxel instead of all N.
        
        Complexity: O(D*H*W * K) instead of O(D*H*W * N)
        
        Args:
            k: Number of nearest Gaussians per voxel
            sigma_cutoff: Zero out contributions beyond this many σ
            batch_size: Process voxels in batches to manage memory
            
        Returns:
            (D, H, W) reconstructed volume
        """
        grid_points = self.grid_points.reshape(-1, 3)  # (D*H*W, 3)
        total_points = grid_points.shape[0]
        
        output = torch.zeros(total_points, device=self.device, dtype=torch.float32)
        
        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            batch_points = grid_points[start:end]
            output[start:end] = self.forward_knn(batch_points, k=k, sigma_cutoff=sigma_cutoff)
        
        return output.reshape(self.D, self.H, self.W)

    def auto_init_scales_from_knn(self, scale_factor: float = 0.5) -> None:
        """
        Initialize scales based on KNN distances (3DGS-style initialization).
        
        Sets each Gaussian's scale proportional to the mean distance to its
        3 nearest neighbors, ensuring reasonable coverage without excessive overlap.
        
        Args:
            scale_factor: Multiplier for KNN distance (0.5 = half the neighbor distance)
        """
        with torch.no_grad():
            mean_dists = self.get_knn_mean_distances()  # (N,)
            
            # Set scales to scale_factor * mean_distance (isotropic)
            target_scales = mean_dists * scale_factor
            target_scales = target_scales.clamp(min=0.001, max=0.3)  # Reasonable bounds
            
            # Convert to raw scale space (inverse softplus)
            # softplus(x) = log(1 + exp(x)), so inverse is: x = log(exp(s) - 1)
            raw_scales = torch.log(torch.exp(target_scales) - 1.0 + 1e-6)
            
            # Set all 3 dimensions to same scale (isotropic)
            self.scales_raw.data = raw_scales.unsqueeze(-1).expand(-1, 3).clone()
            
            print(f"  Auto-initialized scales: min={target_scales.min():.4f}, "
                  f"max={target_scales.max():.4f}, mean={target_scales.mean():.4f}")

    # -------------------- topology edits (densify/clone/split/prune) --------------------
    # IMPORTANT: after any of these, you should re-create your optimiser because nn.Parameters are replaced.

    def _update_gaussians(self, positions_raw, scales_raw, rotations, intensities_raw, raw_opacities):
        self.N = int(positions_raw.shape[0])
        self.positions_raw = nn.Parameter(positions_raw.contiguous())
        self.scales_raw = nn.Parameter(scales_raw.contiguous())
        self.rotations = nn.Parameter(rotations.contiguous())
        self.intensities_raw = nn.Parameter(intensities_raw.contiguous())
        self.raw_opacities = nn.Parameter(raw_opacities.contiguous())
        # Invalidate FAISS cache since Gaussians changed
        self._faiss_index = None
        self._faiss_index_positions = None
    
    def invalidate_knn_cache(self):
        """Call this to force FAISS index rebuild on next KNN query."""
        self._faiss_index = None
        self._faiss_index_positions = None

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float) -> int:
        """Clone Gaussians with high grads and small scales."""
        scales_max = self.scales.max(dim=1).values
        mask = (grads > grad_threshold) & (scales_max <= scale_threshold)
        if mask.sum() == 0:
            return 0

        # small offsets in *raw* space -> stable after sigmoid mapping
        pos_raw = self.positions_raw.data
        new_pos_raw = pos_raw[mask] + torch.randn_like(pos_raw[mask]) * 0.05

        self._update_gaussians(
            positions_raw=torch.cat([self.positions_raw.data, new_pos_raw], dim=0),
            scales_raw=torch.cat([self.scales_raw.data, self.scales_raw.data[mask].clone()], dim=0),
            rotations=torch.cat([self.rotations.data, self.rotations.data[mask].clone()], dim=0),
            intensities_raw=torch.cat([self.intensities_raw.data, self.intensities_raw.data[mask].clone()], dim=0),
            raw_opacities=torch.cat([self.raw_opacities.data, self.raw_opacities.data[mask].clone()], dim=0),
        )
        return int(mask.sum().item())

    def densify_and_split(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float, n_split: int = 2) -> int:
        """Split Gaussians with high grads and large scales into n_split children each."""
        scales_max = self.scales.max(dim=1).values
        mask = (grads > grad_threshold) & (scales_max > scale_threshold)
        if mask.sum() == 0:
            return 0

        idx = mask.nonzero(as_tuple=False).flatten()
        M = idx.numel()

        pos_raw = self.positions_raw.data[idx]
        sc_raw = self.scales_raw.data[idx]
        rot = self.rotations.data[idx]
        inten_raw = self.intensities_raw.data[idx]
        op_raw = self.raw_opacities.data[idx]

        children_pos = []
        children_sc = []
        children_rot = []
        children_int = []
        children_op = []

        # Precompute intensity adjustment for sigmoid activation
        # For sigmoid: to divide output by n_split, we need new_raw = logit(sigmoid(raw)/n)
        if self.intensity_activation == "sigmoid":
            current_intensity = torch.sigmoid(inten_raw)
            target_intensity = current_intensity / n_split
            target_intensity = target_intensity.clamp(1e-6, 1 - 1e-6)  # avoid logit explosion
            # logit = log(p / (1-p))
            new_inten_raw = torch.log(target_intensity / (1 - target_intensity))
        else:
            # For softplus/linear: subtracting log(n) approximately divides output
            new_inten_raw = inten_raw - np.log(n_split + 1e-6)

        for _ in range(n_split):
            # perturb in raw-position space
            children_pos.append(pos_raw + torch.randn_like(pos_raw) * 0.08)
            # reduce scale (in raw scale space, we approximate by subtracting a constant)
            children_sc.append(sc_raw - np.log(n_split + 1e-6))
            children_rot.append(rot.clone())
            children_int.append(new_inten_raw.clone())
            children_op.append(op_raw.clone())

        new_pos_raw = torch.cat(children_pos, dim=0)
        new_sc_raw = torch.cat(children_sc, dim=0)
        new_rot = torch.cat(children_rot, dim=0)
        new_int_raw = torch.cat(children_int, dim=0)
        new_op_raw = torch.cat(children_op, dim=0)

        keep = (~mask)
        self._update_gaussians(
            positions_raw=torch.cat([self.positions_raw.data[keep], new_pos_raw], dim=0),
            scales_raw=torch.cat([self.scales_raw.data[keep], new_sc_raw], dim=0),
            rotations=torch.cat([self.rotations.data[keep], new_rot], dim=0),
            intensities_raw=torch.cat([self.intensities_raw.data[keep], new_int_raw], dim=0),
            raw_opacities=torch.cat([self.raw_opacities.data[keep], new_op_raw], dim=0),
        )
        return int(M)

    def prune_gaussians(self, intensity_threshold: float = 0.01, scale_threshold: float = 0.5) -> int:
        """Prune low-contribution or excessively large Gaussians."""
        scales_max = self.scales.max(dim=1).values
        amps = self.intensities()
        keep = (amps > intensity_threshold) & (scales_max < scale_threshold)

        num_pruned = int((~keep).sum().item())
        if num_pruned == 0:
            return 0

        self._update_gaussians(
            positions_raw=self.positions_raw.data[keep],
            scales_raw=self.scales_raw.data[keep],
            rotations=self.rotations.data[keep],
            intensities_raw=self.intensities_raw.data[keep],
            raw_opacities=self.raw_opacities.data[keep],
        )
        return num_pruned

    def reset_intensity(self, value: float = 0.01) -> None:
        """Optional: clamp or reset negative intensities for linear mode."""
        if self.intensity_activation != "linear":
            return
        with torch.no_grad():
            self.intensities_raw.data = torch.clamp(self.intensities_raw.data, min=value)
