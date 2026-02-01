# VolMicro NeurIPS Paper - Placeholders Summary

This document lists all placeholders (`[PLACEHOLDER]`) in the NeurIPS paper that need to be filled in before submission.

## Author Information
- **Location**: Title/Author block
- **Required**:
  - [ ] Author name(s)
  - [ ] Email address(es)
  - [ ] Department/Lab name
  - [ ] University/Institution
  - [ ] Co-author affiliations (if any)

## Abstract
- No placeholders

## Introduction
- No placeholders

## Related Work
- **Line ~130**: `[PLACEHOLDER: Cite additional INR compression methods if relevant, e.g., COIN++, NeRV, etc.]`
  - Consider adding: COIN++, NeRV, NIRVANA, NeRD, etc.

## Method
- No placeholders

## Experiments

### Experimental Setup
- **Dataset section**: `[PLACEHOLDER: Add additional datasets if available, e.g., other cell types, larger volumes, or public benchmarks]`
  - Consider adding: EM volumes, vasculature data, public BigBrain/Allen Atlas subsets

- **Implementation Details**: `[PLACEHOLDER: Specify exact hardware (GPU memory, CPU), training time, and software versions]`
  - Fill in: GPU model, CUDA version, PyTorch version, training wall-clock time

### Ablation Studies
- **Table 4**: Multiple `[PLACEHOLDER]` entries
  - [ ] Without sparse gating: PSNR, SSIM, Train Time
  - [ ] Without edge weighting: PSNR, SSIM, Train Time  
  - [ ] Without adaptive densification: PSNR, SSIM, Train Time
  - [ ] Fixed N=20K: PSNR, SSIM, Train Time

- **After Table 4**: `[PLACEHOLDER: Describe ablation results and their implications]`

### Training Dynamics Figure
- **Figure 1**: `[PLACEHOLDER: Include training_metrics.pdf figure]`
  - Generate `figures/training_metrics.pdf` with 4 subplots

## Limitations
- **After enumerated list**: `[PLACEHOLDER: Add any additional limitations discovered during experiments]`

## Acknowledgments
- **Entire section**: `[PLACEHOLDER: Acknowledge funding sources, compute resources, and collaborators]`

## Appendix

### Qualitative Results
- **Figure 2**: `[PLACEHOLDER: Visual comparison figure showing MIP projections...]`
  - Generate comparison figure with original, COIN, VolMicro reconstructions

### Broader Impact
- `[PLACEHOLDER: Discuss broader societal impact]`
  - A draft is provided but can be expanded

---

## Figures to Generate

1. **`figures/training_metrics.pdf`**
   - 4-panel figure showing:
     - (a) Total loss (log scale) vs. epoch
     - (b) PSNR vs. epoch
     - (c) MSE (log scale) vs. epoch
     - (d) Number of Gaussians vs. epoch

2. **`figures/qualitative_comparison.pdf`**
   - Side-by-side MIP projections:
     - Original volume
     - COIN reconstruction
     - VolMicro reconstruction
   - Include zoomed insets on dendrite boundaries

3. **`figures/method_overview.pdf`** (optional)
   - Pipeline diagram showing:
     - Input volume → TOPS-Gate → Sparse voxels → Gaussian optimization → Compressed representation

---

## Experiments to Run for Ablations

```bash
# 1. Without sparse gating (train on all voxels)
python train.py --no-sparse-gating --output ablation_no_gating/

# 2. Without edge weighting
python train.py --edge-boost 1.0 --output ablation_no_edge/

# 3. Without adaptive densification (fixed N)
python train.py --no-densification --num-gaussians 20693 --output ablation_fixed_n/

# 4. With fixed N=20K from start
python train.py --no-densification --num-gaussians 20000 --output ablation_20k/
```

---

## Checklist Before Submission

- [ ] Fill in all author information
- [ ] Run ablation experiments and fill in Table 4
- [ ] Generate training metrics figure
- [ ] Generate qualitative comparison figure
- [ ] Write ablation analysis paragraph
- [ ] List specific hardware and software versions
- [ ] Add additional datasets (if available)
- [ ] Write acknowledgments
- [ ] Expand broader impact statement
- [ ] Compile and check for formatting issues
- [ ] Check page limit (NeurIPS: 9 pages main + unlimited references/appendix)
- [ ] Run spell check
- [ ] Verify all citations are correct
