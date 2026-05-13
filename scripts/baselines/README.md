# External Baseline Wrappers

## AdvDiff

`run_advdiff_manifest.py` bridges the platform's external attack protocol to
the third-party `EricDai0/advdiff` implementation vendored as a git submodule
under `third_party/advdiff`.

AdvDiff is class-conditional unrestricted generation. It uses the manifest
labels to generate one image per row; it does not edit the row's input image.
This makes it useful as a strong generative baseline, but its LPIPS-to-original
semantics differ from img2img attacks.

Server setup:

```bash
git submodule update --init --recursive third_party/advdiff
cd third_party/advdiff
conda env create -f environment.yaml
conda activate ldm_adv
mkdir -p models/ldm/cin256-v2
wget -O models/ldm/cin256-v2/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
```

Run from the main project environment:

```bash
export ADVDIFF_PYTHON=/path/to/conda/envs/ldm_adv/bin/python
.venv/bin/python -m src.cli run configs/paper/07_advdiff_external.yaml
```

Useful knobs:

```bash
export ADVDIFF_BATCH_SIZE=4
export ADVDIFF_DDIM_STEPS=50
export ADVDIFF_K=2
```

Increase `ADVDIFF_DDIM_STEPS` and `ADVDIFF_K` for stronger but slower runs.
