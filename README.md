# CoRe: Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting

This is the official code repository for the paper **CoRe: Cross-variate Correction-space Interaction for Test-time Adaptation in Time Series Forecasting**.

## Overview

CoRe is a lightweight test-time adaptation framework for multivariate time series forecasting. It augments a frozen pretrained backbone with a learnable adapter that predicts residual corrections, refined via Shared-shift Correction Refinement (SCR), a structured cross-variate interaction module operating in the correction space.

## Quick Start

```bash
Extract the downloaded archive and follow the installation instructions below.

cd CoRe-TTA

pip install -r requirements.txt

python main.py \
  DATA.NAME exchange_rate \
  DATA.PRED_LEN 720 \
  MODEL.NAME DLinear \
  MODEL.pred_len 720 \
  TRAIN.ENABLE False \
  TRAIN.CHECKPOINT_DIR ./checkpoints/DLinear/exchange_rate_720/ \
  TTA.ENABLE True \
  RESULT_DIR results/CORE/ \

  TTA.COSA.STEPS 20 \
  TTA.CORE.GATE_BIAS_INIT -1.0
```

Expected output: `adapt_test_mse=0.1213, adapt_test_mae=0.2487`

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

Download datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange-Rate, Traffic, Electricity, Traffic, Electricity) and place them under `./data/`.

## Usage

### Training a backbone

```bash
python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME DLinear \
  TRAIN.ENABLE True \
  TRAIN.CHECKPOINT_DIR ./checkpoints/DLinear/ETTh1_96/
```

### Test-time adaptation with CoRe

After training a backbone, run TTA with CoRe (example for DLinear on ETTh1, pred_len=96):

```bash
python main.py \
  DATA.NAME ETTh1 \
  DATA.PRED_LEN 96 \
  MODEL.NAME DLinear \
  MODEL.pred_len 96 \
  TRAIN.ENABLE False \
  TRAIN.CHECKPOINT_DIR ./checkpoints/DLinear/ETTh1_96/ \
  TTA.ENABLE True \
  RESULT_DIR results/CORE/ \

  TTA.COSA.STEPS 20 \
  TTA.CORE.GATE_BIAS_INIT -1.0
```

The result directory will be automatically set to `results/CORE/DLinear/ETTh1_96/`. For a ready-to-run example without training, see Quick Start above.

### Baselines

Replace the RESULT_DIR prefix with TAFAS, TAFAS_CORE, PETSA, DYNATTA, or COSA to run the corresponding baseline.

## Supported Backbones

DLinear, iTransformer, PatchTST, FreTS, MICN, OLS, Informer

## Supported Datasets

ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange-Rate, Traffic, Electricity

## License

See LICENSE for details.

## Acknowledgements

This codebase is built upon the following open-source projects:

- **TAFAS** ([Kim et al., 2025](https://arxiv.org/abs/2501.04970)) — the overall codebase structure, data loading, model training, and configuration system are adapted from the [TAFAS repository](https://github.com/kimanki/TAFAS), licensed under the Modified MIT License (Non-Commercial with Permission).
- **COSA** ([Im and Kwon, 2026](https://openreview.net/forum?id=L7Z5wBMPrW)) and **TAFAS** serve as base adapters in our comparison framework.

We thank the authors for making their code publicly available.
