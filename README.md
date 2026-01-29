# TOPS-GATE

> **T**omographic **O**ptical **P**rojection with **S**parse **G**aussian **A**daptive **T**omography **E**ncoding

Gaussian-based implicit neural representation for 3D volume data compression and reconstruction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TOPS-GATE represents volumetric data as a weighted sum of 3D Gaussian basis functions, enabling efficient compression while maintaining high reconstruction quality:

$$f(x, y, z) = \sum_{i=1}^{N} w_i \cdot G_i(x, y, z; \mu_i, \Sigma_i)$$

where each Gaussian is defined as:

$$G_i(\vec{x}; \mu_i, \Sigma_i) = \exp\left\{-\frac{1}{2}(\vec{x} - \mu_i)^T \Sigma_i^{-1} (\vec{x} - \mu_i)\right\}$$

## Features

- 🚀 **CUDA-accelerated** training and inference
- 📦 **High compression ratios** with minimal quality loss
- 🎛️ **Adaptive Gaussian placement** for sparse representations
- 📊 **Flexible regularization** (sparsity, overlap, smoothness)
- 🔧 **Multiple training backends** (standard, fast, CUDA-optimized)

---

## Installation

```bash
git clone https://github.com/your-org/TOPS-GATE.git
cd TOPS-GATE
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.0 (for GPU acceleration)
- NumPy, tifffile, PyYAML

---

## Quick Start

### Basic Training

```bash
python train.py --volume path/to/volume.tif --num_gaussians 5000 --epochs 100
```

### CUDA-Optimized Training

```bash
python train_cuda.py --volume path/to/volume.tif --num_gaussians 10000
```

### Full Configuration

```bash
python train.py \
    --volume path/to/volume.tif \
    --num_gaussians 10000 \
    --init_method uniform \
    --epochs 200 \
    --batch_size 8192 \
    --lr 0.01 \
    --optimizer adam \
    --lambda_sparsity 0.001 \
    --lambda_overlap 0.001 \
    --lambda_smoothness 0.001 \
    --output_dir results \
    --device cuda
```

---

## Project Structure

```
TOPS-GATE/
├── gaussian_model.py       # Core Gaussian model (CPU)
├── gaussian_model_cuda.py  # CUDA-optimized model
├── losses.py               # Loss functions & regularization
├── trainer.py              # Training loop
├── train.py                # Standard training script
├── train_cuda.py           # CUDA training script
├── train_fast.py           # Optimized training script
├── train_with_gate_sparse.py  # Sparse gating training
├── load_model.py           # Model loading utilities
├── render_with_gsplat.py   # Rendering with gsplat
├── config.yml              # Default configuration
└── gs_utils/               # Utility functions
    ├── Compute_intensity.py
    ├── general_utils.py
    └── discretize_grid.cu
```

---

## Algorithm

### Optimization Objective

$$\min_{\{N, \mu_i, \Sigma_i, w_i\}} \sum_{k=1}^{M} \left\| v_k - \sum_{i=1}^{N} w_i \cdot G_i(\vec{x}_k; \mu_i, \Sigma_i) \right\|^2 + \mathcal{R}$$

### Regularization Terms

| Term | Formula | Purpose |
|------|---------|---------|
| **Sparsity** | $\lambda_w \sum_i \|w_i\|$ | Encourage minimal weights |
| **Overlap** | $\lambda_o \sum_{i \neq j} \text{overlap}(G_i, G_j)$ | Reduce redundancy |
| **Smoothness** | $\lambda_s \sum_i \|\nabla_\mu G_i\|^2$ | Spatial coherence |

### Training Pipeline

1. **Initialize** Gaussian parameters ($\mu_i$, $\Sigma_i$, $w_i$)
2. **Sample** voxel coordinates from volume
3. **Forward** pass through Gaussian mixture
4. **Compute** loss with regularization
5. **Backpropagate** and update parameters
6. **Repeat** until convergence

---

## Python API

```python
import torch
from gaussian_model import GaussianVolumeModel
from trainer import Trainer

# Load volume data
volume = torch.from_numpy(tifffile.imread('volume.tif')).float()

# Initialize model
model = GaussianVolumeModel(
    num_gaussians=5000,
    volume_shape=volume.shape,
    device='cuda'
)

# Create trainer
trainer = Trainer(
    model=model,
    volume=volume,
    learning_rate=0.01,
    lambda_sparsity=0.001
)

# Train
history = trainer.train(epochs=100, batch_size=4096)

# Reconstruct
reconstructed = model.reconstruct_volume()
```

---

## Outputs

| File | Description |
|------|-------------|
| `checkpoints/*.pt` | Model checkpoints |
| `reconstructed.tif` | Reconstructed volume |
| `history.json` | Training metrics |
| `gaussian_parameters.npz` | Learned Gaussian parameters |

---

## Configuration

See [config.yml](config.yml) for all available options:

```yaml
model:
  num_gaussians: 5000
  init_method: uniform

training:
  epochs: 100
  batch_size: 8192
  learning_rate: 0.01
  optimizer: adam

regularization:
  lambda_sparsity: 0.001
  lambda_overlap: 0.001
  lambda_smoothness: 0.001
```

---

## Citation

```bibtex
@software{tops_gate2024,
  title = {TOPS-GATE: Gaussian-Based Volume Representation},
  year = {2024},
  url = {https://github.com/your-org/TOPS-GATE}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
# VolMicro
