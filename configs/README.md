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

Run the canonical paper suite:

```bash
python -m src.cli run-batch configs/paper
```

## `ablations/`

Focused experiments for Chapter 5 analysis.

| Directory | Purpose |
|-----------|---------|
| `defense/` | Compare no defense, JPEG, blur, bit-depth, and diffusion purification |
| `diffusion_strength/` | Analyze attack strength versus ASR and visual/semantic quality |
| `purification_steps/` | Analyze diffusion purification steps versus robust accuracy, clean drop, and latency |

## `smoke/`

Small, cheap checks for development. These are not thesis evidence.

```bash
python -m src.cli run configs/smoke/synthetic.yaml
python -m src.cli run configs/smoke/cpu.yaml
python -m src.cli run configs/smoke/sd_tiny_e2e.yaml
```

