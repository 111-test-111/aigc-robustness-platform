# Experiment Configuration Map

This directory is organized by purpose so paper experiments, ablations, and
development smoke checks do not get mixed together.

## `paper/`

Canonical experiments for the thesis.

| File | Purpose | Expected use |
|------|---------|--------------|
| `01_traditional_attack_baseline.yaml` | FGSM/PGD on ResNet-50 with JPEG, blur, and bit-depth defenses | Traditional baseline table |
| `02_generative_attack_mainline.yaml` | Stable Diffusion img2img attack with JPEG and diffusion purification | Main AIGC/unrestricted attack result |
| `03_full_resnet50_suite.yaml` | FGSM, PGD, and diffusion attack with all implemented defenses | Final comprehensive table when compute budget allows |
| `04_cross_model_A_traditional.yaml` | FGSM, PGD, and generator baseline on ResNet-50 with lightweight defenses | Cross-model traditional phase template |
| `04_cross_model_B_purification.yaml` | FGSM, PGD, and generator baseline with diffusion purification | Cross-model purification phase template |
| `04_cross_model_C_diffusion.yaml` | Diffusion attack with all defenses | Cross-model diffusion phase template |
| `05_targeted_attack.yaml` | Untargeted vs targeted diffusion attack comparison | Targeted attack analysis |
| `06_strong_linf_autoattack.yaml` | AutoAttack on ResNet-50 with lightweight defenses | Strong Lp-bounded baseline comparison |
| `07_advdiff_external.yaml` | External AdvDiff class-conditional generation baseline | Strong unrestricted generative baseline comparison |

Run the canonical paper suite:

```bash
python -m src.cli run-batch configs/paper
```

## `ablations/`

Focused experiments for Chapter 5 analysis.

### `attack/` — Attack parameter sweeps

| Directory | Parameters varied | Values |
|-----------|-------------------|--------|
| `fgsm_eps/` | eps | 0.01, 0.03, 0.08, 0.15 |
| `pgd_eps/` | eps | 0.01, 0.03, 0.08, 0.15 |
| `pgd_steps/` | steps | 7, 20, 40 |
| `advgan/` | eps, epochs | eps ∈ {0.03, 0.08}, epochs ∈ {25, 50, 100} |
| `diffusion/` | strength, guidance_scale, num_candidates | strength ∈ {0.3, 0.5, 0.7}, guidance ∈ {3.0, 7.5, 15.0}, candidates ∈ {1, 3, 5, 10} |

### `defense/` — Defense parameter sweeps

| Directory | Parameters varied | Values |
|-----------|-------------------|--------|
| `jpeg_quality/` | quality | 25, 50, 75, 90 |
| `blur_config/` | kernel_size, sigma | (3,0.5), (5,1.0), (7,1.5), (9,2.0) |
| `bit_depth/` | bits | 2, 3, 4, 5, 6 |
| `diffusion_purification/` | steps, noise_level | steps ∈ {10, 20, 50}, noise ∈ {0.05, 0.10, 0.20} |
| `defense_00_no_defense.yaml` | No defense baseline | Reference point for all defense ablations |

### `cross/` — Attack × Defense interaction matrix

Each config tests one attack against all applicable defenses, measuring
which defense works best for each attack type.

| File | Attack | Defenses tested |
|------|--------|-----------------|
| `fgsm_vs_defenses.yaml` | FGSM | JPEG, Gaussian Blur, Bit Depth |
| `pgd_vs_defenses.yaml` | PGD | JPEG, Gaussian Blur, Bit Depth |
| `advgan_vs_defenses.yaml` | AdvGAN | JPEG, Gaussian Blur, Bit Depth |
| `diffusion_vs_defenses.yaml` | Diffusion | JPEG, Gaussian Blur, Bit Depth, Diffusion Purification |

### `model/` — Model architecture ablations

| File | Model | Purpose |
|------|-------|---------|
| `vit_b_16_A_traditional.yaml` | ViT-B/16 | Transformer vs CNN robustness under traditional/generator attacks |
| `vit_b_16_B_purification.yaml` | ViT-B/16 | Transformer response to diffusion purification |
| `vit_b_16_C_diffusion.yaml` | ViT-B/16 | Transformer response to diffusion attack |
| `densenet121_A_traditional.yaml` | DenseNet121 | Dense connectivity robustness under traditional/generator attacks |
| `densenet121_B_purification.yaml` | DenseNet121 | Dense connectivity response to diffusion purification |
| `densenet121_C_diffusion.yaml` | DenseNet121 | Dense connectivity response to diffusion attack |

## `smoke/`

Small, cheap checks for development. These are not thesis evidence.

```bash
python -m src.cli run configs/smoke/synthetic.yaml
python -m src.cli run configs/smoke/cpu.yaml
python -m src.cli run configs/smoke/sd_tiny_e2e.yaml
```

## `templates/`

Templates for external strong baselines that are not vendored into this repo.
Copy one into `configs/paper/` after filling in the external command path.

| File | Purpose |
|------|---------|
| `advdiffuser_external.yaml` | Runs an external AdvDiffuser-compatible implementation through the manifest protocol |
| `advdiff_external.yaml` | Runs an external AdvDiff or AdvDiff-style guided diffusion implementation |

## Experiment Dimensions Summary

The config matrix covers these dimensions for comprehensive robustness evaluation:

| Dimension | Values in configs |
|-----------|-------------------|
| **Attack methods** | FGSM, PGD, AutoAttack, generator baseline, DiffusionAttack, external AdvDiffuser/AdvDiff adapters |
| **Defense methods** | JPEG, Gaussian Blur, Bit Depth, Diffusion Purification |
| **Model architectures** | ResNet50, ViT-B/16, DenseNet121 |
| **Attack strength (eps)** | 0.01, 0.03, 0.08, 0.15 |
| **PGD iterations** | 7, 20, 40 |
| **Diffusion strength** | 0.3, 0.5, 0.7 |
| **Diffusion guidance** | 3.0, 7.5, 15.0 |
| **Diffusion candidates** | 1, 3, 5, 10 |
| **JPEG quality** | 25, 50, 75, 90 |
| **Blur (kernel, sigma)** | (3,0.5), (5,1.0), (7,1.5), (9,2.0) |
| **Bit depth** | 2, 3, 4, 5, 6 |
| **Purification steps** | 10, 20, 50 |
| **Purification noise** | 0.05, 0.10, 0.20 |
| **Attack type** | Untargeted, Targeted |
| **Sample size** | 200 (diffusion), 500 (traditional) |
| **Seeds** | 5 seeds (42, 123, 456, 789, 1024) for error bars |
