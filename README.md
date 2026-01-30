# VolMicro

> **3D Gaussian Mixtures for Microscopy Volume Compression**  
> Compact Neural Representations for Large-Scale Biological Imaging Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Modern light-sheet and confocal microscopy generates terabyte-scale volumetric datasets that pose severe challenges for storage, transmission, and analysis. **VolMicro** is a compression framework based on 3D Gaussian mixture models that achieves **extreme compression ratios (>100×)** while preserving the fine structural details critical for biological analysis.

Our method represents microscopy volumes as sparse collections of anisotropic 3D Gaussians, where each Gaussian encodes local intensity with learnable position, scale, orientation, and amplitude.

### Key Innovations

- 🎯 **Sparse Occupancy Gating**: Restricts computation to biologically relevant regions (typically <15% of volume), achieving **10-20× training speedup**
- 🔬 **Edge-Aware Loss Weighting**: Preserves dendrite boundaries and fine neurite structures
- 📈 **Adaptive Densification**: Automatically places more Gaussians in regions of high structural complexity
- ⚡ **CUDA-Accelerated KNN Evaluation**: Enables real-time decompression

### Performance

Benchmark on neuron microscopy volume (100 × 647 × 813 voxels, 16-bit):

| Metric | Result |
|--------|--------|
| **PSNR** | 36.56 dB |
| **SSIM** | 0.932 |
| **LPIPS** | 0.278 |
| **Compression Ratio** | 231× |
| **bpp** | 0.073 |
| **Model Size** | 0.46 MB (20,693 Gaussians) |
| **Training Time** | 7 min (10K epochs) |

---

## Method

### Gaussian Primitive Field Representation

We represent the volume as a finite mixture of anisotropic 3D Gaussian primitives:

$$\hat{V}(\mathbf{x};\Theta) = \sum_{i=1}^{N} w_i \cdot \phi(\mathbf{x}; \mu_i, \Sigma_i)$$

where:
- $w_i \in \mathbb{R}$ is the scalar amplitude (intensity)
- $\mu_i \in \Omega$ is the Gaussian center position
- $\Sigma_i \in \mathbb{R}^{3\times3}$ is the covariance matrix (size and orientation)

The Gaussian kernel is:

$$\phi(\mathbf{x}; \mu, \Sigma) = \exp\left(-\frac{1}{2}(\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right)$$

### Sparsity-Aware Modeling

Microscopy volumes are background-dominant. We exploit this by restricting training to an **occupied set** $\mathcal{O}$ obtained from a learned occupancy gate:

$$\mathcal{O} = \{\mathbf{x} \in \Omega : g(\mathbf{x}) \geq \tau\}$$

This reduces unnecessary modeling of empty background and improves rate-distortion efficiency.

### Structure-Preserving Objective

$$\mathcal{L}(\Theta) = \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\nabla}\mathcal{L}_{\nabla} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$$

- **Weighted Reconstruction Loss**: Edge-aware intensity loss with higher penalty near high-frequency content
- **Gradient Consistency**: Matches spatial derivatives to preserve topology-relevant edges
- **Regularization**: Prevents degenerate primitives and encourages parsimonious representations

---

## Installation

```bash
git clone https://github.com/asheibanifard/VolMicro.git
cd VolMicro
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.0 (for GPU acceleration)
- FAISS (for KNN acceleration)
- NumPy, tifffile, PyYAML

---

## Pretrained Models

Download pretrained models from HuggingFace:

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download all pretrained models
huggingface-cli download Arminshfard/volmicro-checkpoints --local-dir ./pretrained_models
```

Or download directly from: 🤗 [Arminshfard/volmicro-checkpoints](https://huggingface.co/Arminshfard/volmicro-checkpoints)

### Available Models

| Model | Gaussians | PSNR | Compression | Description |
|-------|-----------|------|-------------|-------------|
| `v019` | 20,693 | 36.56 dB | 231× | Best quality, 10K epochs |
| `v014` | -- | -- | -- | Ablation study |

### Quick Inference

```python
from load_model import load_checkpoint

# Load pretrained model
model = load_checkpoint('pretrained_models/v019/checkpoints/checkpoint_epoch_010000.pt')

# Reconstruct volume
volume = model.forward_knn_volume(k=32)
print(f"Reconstructed: {volume.shape}")  # (100, 647, 813)
```

---

## Quick Start

### 1. Train Occupancy Gate (TOPS-Gate)

First, train the sparse occupancy gate on your volume:

```bash
python train_gate.py --volume path/to/volume.tif --output outputs/TOPS_GATE
```

### 2. Train Gaussian Compression Model

```bash
python train.py \
    --volume path/to/volume.tif \
    --gate_checkpoint outputs/TOPS_GATE/tops_gate_step_020000.pt \
    --gate_tau 0.5 \
    --num_gaussians 15000 \
    --epochs 5000 \
    --lr 0.01 \
    --output_dir outputs/compressed \
    --densify \
    --max_gaussians 50000 \
    --use_sampling \
    --num_samples 300000 \
    --grad_threshold 0.00002
```

### 3. Decompress / Reconstruct

```python
from load_model import load_checkpoint

# Load compressed representation
model = load_checkpoint('outputs/compressed/checkpoints/checkpoint_epoch_005000.pt')

# Reconstruct full volume
reconstructed = model.forward()  # (D, H, W) tensor

# Region-of-interest query (efficient!)
roi = model.forward_knn(query_points, k=32)
```

---

## Storage Format

The compressed representation stores per-Gaussian parameters:

| Parameter | Size |
|-----------|------|
| Position | 3 × float16 = 6 bytes |
| Scale | 3 × float16 = 6 bytes |
| Rotation | 4 × float16 = 8 bytes |
| Intensity | 1 × float16 = 2 bytes |
| **Total** | **22 bytes per Gaussian** |

**Example**: Neuron microscopy volume (100 × 647 × 813, 105 MB at 16-bit) with 20,693 Gaussians:

$$\text{Compression ratio} = \frac{105.2\text{ MB}}{20693 \times 22\text{ bytes}} = \frac{105.2\text{ MB}}{0.46\text{ MB}} \approx 231\times$$

---

## Comparison with Baselines

Benchmark on neuron microscopy volume (100 × 647 × 813 voxels, 16-bit, 105 MB):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | bpp ↓ | Ratio ↑ | Size | Time |
|--------|--------|--------|---------|-------|---------|------|------|
| JPEG2000-3D | 41.09 dB | 0.935 | 0.204 | 0.333 | 101× | 2.09 MB | 0.06 min |
| HEVC (x265) | 41.62 dB | 0.943 | 0.015 | 0.316 | 106× | 1.98 MB | 0.04 min |
| ZFP (ε=10⁻³) | 87.45 dB | 1.000 | 0.000 | 10.23 | 3.3× | 64.12 MB | 0.03 min |
| SIREN | 33.1 dB | 0.931 | -- | 0.209 | 80× | 1.31 MB | 25 min |
| Dense Gaussian | 34.8 dB | 0.958 | -- | 0.177 | 95× | 1.11 MB | 42 min |
| **VolMicro (Ours)** | **36.56 dB** | **0.932** | **0.278** | **0.073** | **231×** | **0.46 MB** | **7 min** |

> **Note**: PSNR/SSIM computed on gated (foreground) region. ZFP achieves near-lossless quality but at 70× larger size than VolMicro. Traditional codecs (JPEG2000, HEVC) have higher PSNR but worse compression ratios.

---

## Project Structure

```
VolMicro/
├── train.py                    # Main training script (sparse gate-guided)
├── train_cuda.py               # CUDA-optimized training
├── gaussian_model_cuda.py      # CUDA Gaussian model with KNN
├── gaussian_model.py           # CPU Gaussian model
├── losses.py                   # Loss functions & regularization
├── trainer.py                  # Training utilities
├── load_model.py               # Checkpoint loading
├── export_to_splat.py          # Export to .splat format
├── config.yml                  # Default configuration
├── gs_utils/
│   ├── Compute_intensity.py    # CUDA intensity computation
│   ├── discretize_grid.cu      # CUDA kernel
│   └── general_utils.py        # Utilities
└── docs/
    └── paper.pdf               # Technical paper
```

---

## Key Features

### Adaptive Densification

Automatically splits and clones Gaussians based on gradient magnitude:

```bash
--densify                    # Enable densification
--grad_threshold 0.00002     # Gradient threshold for clone/split
--max_gaussians 50000        # Maximum Gaussians
```

### Edge-Aware Loss

Preserves thin neurite structures by weighting loss higher on edges:

```bash
--edge_boost 3.0             # Edge weight multiplier
--no_edge_weights            # Disable edge weighting
```

### Sparse Gating

Exploits background sparsity for 6-9× speedup:

```bash
--gate_checkpoint path.pt    # TOPS-Gate checkpoint
--gate_tau 0.5               # Occupancy threshold
```

---

## Advantages for Microscopy Workflows

### Progressive Loading
The continuous Gaussian representation enables multi-resolution queries. Quickly preview low-resolution reconstructions before loading full detail.

### Region-of-Interest Extraction
Unlike block-based codecs, our representation supports efficient spatial queries. Extracting a subvolume requires evaluating only nearby Gaussians.

### Semantic Alignment
Gaussians naturally align with biological structures. Elongated Gaussians follow dendrite axes; spherical Gaussians capture somata.

---

## Citation

```bibtex
@article{volmicro2026,
  title={3D Gaussian Mixtures for Microscopy Volume Compression: 
         Compact Neural Representations for Large-Scale Biological Imaging Data},
  author={Sheibanifard, Armin},
  year={2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds upon advances in 3D Gaussian Splatting for novel-view synthesis and extends them to the domain of volumetric microscopy compression.
