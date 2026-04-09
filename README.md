# MRI Modality Translation

Clean PyTorch baseline repository for paired MRI modality translation on 2D slices.

This repository refactors a notebook-based MSc project into a modular experiment layout. The baseline task is supervised **T1 → T2 MRI slice translation** using paired `64×64` grayscale images from the Hugging Face dataset `dpelacani/mri-t1-t2-2D-sliced-64`.

## Scope

What this repository is:
- a reproducible PyTorch baseline for paired image-to-image regression
- a comparison between a simple encoder-decoder and a U-Net
- a clean starting point for controlled evaluation and later extensions

What this repository is not:
- a clinical system
- a deployment-ready medical product
- evidence of diagnostic utility or real-world hospital performance

## Repository layout

```text
configs/                experiment configs
scripts/                CLI entry points
src/mri_translation/    reusable package code
tests/                  unit tests for core logic
notebooks/              light EDA and results notebooks
reports/figures/        exported figures for README/report
outputs/runs/           checkpoints, histories, metrics
```

## Baseline workflow

1. Install dependencies
2. Run a baseline training config
3. Evaluate the saved checkpoint
4. Review saved metrics and plots
5. Iterate on one change at a time

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/train.py --config configs/baseline_encoder_decoder.yaml
python scripts/train.py --config configs/unet.yaml

python scripts/evaluate.py --config configs/eval.yaml --checkpoint outputs/runs/baseline_encoder_decoder/best.pt
python scripts/evaluate.py --config configs/eval.yaml --checkpoint outputs/runs/unet/best.pt
```

## Data

Default dataset:
- `dpelacani/mri-t1-t2-2D-sliced-64`

Expected fields:
- `t1`
- `t2`
- `view`
- `patient_id`

## Current modelling choices

- input: single-channel T1 slice
- target: single-channel T2 slice
- objective: paired supervised regression
- losses: MSE by default
- models:
  - simple encoder-decoder baseline
  - U-Net baseline
- evaluation:
  - per-pixel MSE / MAE / RMSE
  - PSNR
  - SSIM
  - qualitative prediction grids

## Notes on normalisation

The original notebook used global min-max scaling estimated from a sampled pass over the training set. This repository keeps normalisation configurable so the notebook-faithful baseline can be reproduced first, then alternatives such as percentile-based scaling can be tested cleanly.

## Recommended refactor sequence

1. reproduce notebook behaviour in modular code
2. retrain both baseline models
3. clean up evaluation and checkpointing
4. only then test extensions such as alternative normalisation or more stable U-Net settings

## Limitations

- evaluation is currently on the provided validation split only
- results are slice-based, not 3D volume-based
- pixel-wise metrics do not fully capture perceptual quality
- stronger medical or clinical claims are not supported by this baseline alone
