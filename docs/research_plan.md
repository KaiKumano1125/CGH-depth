# Physics-Informed Deep Learning CGH Pipeline Plan

## Objective
Build a physics-informed deep learning pipeline for the KOREATECH-CGH dataset (6,000 RGB-D/Hologram pairs; depth range up to 80 mm; resolutions up to 2048×2048) that surpasses the AP-LBM baseline (23.99 dB PSNR) by coupling learned representations with propagation physics.

## Research Timeline

### Phase 1 — Feature Engineering (Weeks 1–4)
**Goal:** Provide the network with explicit physics-aware features that encode depth-dependent diffraction and improve layer discrimination.

1. **Depth-Frequency Encoding**
   - Encode propagation distance z into spatial-frequency features, enabling the model to learn diffraction transfer functions.
   - Represent z as a frequency-domain scaling/phase feature derived from the angular spectrum method (ASM) grid.
   - Verify that the encoding correlates with known diffraction behavior by inspecting frequency-energy distribution shifts.

2. **Layer-Specific Masking (Feature Buckets)**
   - Pre-split RGB intensities into 8 depth-bucket channels covering the 0–80 mm range.
   - Combine buckets with depth-frequency encoding to form a multi-channel input.

3. **Physics-Driven Loss (Focal Image Projection)**
   - Implement FIP logic to aggregate in-focus regions across a focal stack.
   - Enforce sharpness and energy preservation by penalizing off-focus leakage within each depth bucket.

**Deliverables:** Feature engineering code, data loaders with bucketized channels, and a unit-testable physics loss module.

---

### Phase 2 — Hybrid Swin-Transformer + ASM Architecture (Weeks 5–9)
**Goal:** Build a dual-path network that fuses global context with explicit propagation physics.

1. **Hybrid Backbone**
   - Swin-Transformer block for global depth-context modeling.
   - CNN/UNet stem for local texture and hologram feature refinement.

2. **Propagation Module**
   - ASM-based propagation layer for differentiable wavefield projection.
   - Depth-frequency encoding injected before propagation to guide the network.

3. **Training Protocol**
   - Curriculum from low-resolution crops to full-resolution 2048×2048.
   - Mixed precision + gradient checkpointing for large-scale holograms.

**Deliverables:** Model skeleton and training script with differentiable ASM.

---

### Phase 3 — Benchmarking and Evaluation (Weeks 10–12)
**Goal:** Validate performance relative to AP-LBM and quantify improvements.

1. **Metrics**
   - PSNR, SSIM, and depth-aware fidelity (per bucket).

2. **Baselines**
   - AP-LBM (target > 23.99 dB PSNR).
   - Ablation studies: remove depth-frequency encoding, remove FIP loss, remove bucket channels.

3. **Reporting**
   - Reproduce AP-LBM settings where possible to ensure fair comparison.
   - Generate qualitative focal stack comparisons.

**Deliverables:** Benchmark report with plots and ablation tables.

## Data Flow Summary
1. **Input:** RGB image + depth map.
2. **Feature Engineering:**
   - 8-channel depth buckets.
   - depth-frequency encoding channels from z and frequency grid.
3. **Model:** Hybrid Swin-Transformer + ASM propagation head.
4. **Loss:** FIP-based physics loss + reconstruction losses.

## Risk Mitigation
- **Large resolution memory pressure:** Use patch-based training and gradient checkpointing.
- **Depth bucket imbalance:** Use stratified sampling or per-bucket weighting.
- **Propagation mismatch:** Validate ASM parameters with synthetic phantoms.
