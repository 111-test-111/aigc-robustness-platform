# External Baseline Wrappers

## AdvDiff

`run_advdiff_manifest.py` bridges the platform's external attack protocol to
the third-party `EricDai0/advdiff` implementation vendored as source files
under `third_party/advdiff`.

AdvDiff is class-conditional unrestricted generation. It uses the manifest
labels to generate one image per row; it does not edit the row's input image.
This makes it useful as a strong generative baseline, but its LPIPS-to-original
semantics differ from img2img attacks.

Server setup from the repository root:

```bash
conda env create -f scripts/baselines/environment_advdiff_a800.yml
conda activate advdiff-a800
mkdir -p third_party/advdiff/models/ldm/cin256-v2
wget -O third_party/advdiff/models/ldm/cin256-v2/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
```

The original AdvDiff environment pins `torch=1.7.0`. For A800/Ampere servers,
prefer `environment_advdiff_a800.yml`, which uses PyTorch 2.1.2 with CUDA 11.8.
The wrapper installs inference-only compatibility shims for old
`pytorch_lightning.utilities.distributed` imports, so the vendored AdvDiff
source can stay unchanged.
The environment file avoids GitHub-only pip dependencies; the vendored
AdvDiff tree includes the minimal `taming` and `clip` import shims needed for
the ImageNet class-conditional inference path.

Run from the main project environment:

```bash
export ADVDIFF_PYTHON=/path/to/conda/envs/advdiff-a800/bin/python
.venv/bin/python -m src.cli run configs/paper/07_advdiff_external.yaml
```

Useful knobs:

```bash
export ADVDIFF_BATCH_SIZE=4
export ADVDIFF_DDIM_STEPS=50
export ADVDIFF_K=2
```

Increase `ADVDIFF_DDIM_STEPS` and `ADVDIFF_K` for stronger but slower runs.
