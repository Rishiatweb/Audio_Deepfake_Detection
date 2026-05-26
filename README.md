# ConDetection-DANN: Cross-Domain Audio Deepfake Detection

A modular PyTorch implementation of a hierarchical multi-scale Conformer with Domain-Adversarial Neural Network (DANN) for generalizable audio deepfake detection.

---

## Overview

Existing audio deepfake detectors degrade severely when deployed outside their training distribution. Müller et al. (2022) showed up to **1000% EER increase** when moving from controlled benchmarks to real-world audio. ConDetection-DANN directly addresses this via:

- **Multi-scale Conformer** — three spectral resolutions (fine/mid/coarse) processed in parallel
- **Cross-Scale Attention Fusion** — learned inter-resolution attention weights
- **Domain-Adversarial Training (DANN)** — gradient reversal forces domain-invariant feature learning
- **MixStyle** — feature statistics mixing for domain generalization

Trained on **Fake or Real (FoR)**, evaluated on **In-the-Wild (ITW)** as hard out-of-domain benchmark.

---

## Architecture

```
Audio → [Fine Mel | Mid Mel | Coarse Mel]
             ↓           ↓          ↓
       ScaleEncoder × 3  (CNN → projection)
             ↓           ↓          ↓
       ConformerBlocks × 2 (shared)
             ↓           ↓          ↓
       CrossScaleAttentionFusion
             ↓
       Classifier (real/fake)
             ↓
       DomainDiscriminator ← GradientReversal
```

| Scale  | n_fft | hop_length | n_mels | d_model |
|--------|-------|------------|--------|---------|
| Fine   | 400   | 160        | 64     | 128     |
| Mid    | 1024  | 256        | 80     | 128     |
| Coarse | 2048  | 512        | 128    | 128     |

**Parameters:** ~1.56M trainable

---

## Results

### SOTA Comparison (FoR test + ITW out-of-domain)

| Model | Params | FoR EER↓ | FoR AUC↑ | ITW EER↓ | ITW AUC↑ | Gen. Gap↓ |
|-------|--------|----------|----------|----------|----------|-----------|
| LCNN | 0.17M | 0.1376 | 0.9444 | 0.3054 | 0.7746 | 0.1678 |
| AASIST | 0.82M | 0.0367 | 0.9944 | 0.2755 | 0.8056 | 0.2388 |
| **ConDetection-DANN** | **1.56M** | **0.0516** | **0.9826** | **0.1951** | **0.8930** | **0.1436** |

ConDetection-DANN achieves the best out-of-domain generalization (lowest ITW EER, lowest gen gap) despite not having the best in-domain score.

### Ablation Study

| Variant | FoR EER↓ | ITW EER↓ | Gen Gap↓ | disc_acc | \|dev from 0.5\| |
|---------|----------|----------|----------|----------|-----------------|
| Full model | 0.0725 | 0.1902 | 0.1177 | 0.587 | 0.087 |
| No DANN | 0.0481 | 0.2808 | 0.2327 | N/A | N/A |
| Single scale (mid) | 0.0563 | 0.2952 | 0.2389 | 0.604 | 0.104 |
| No MixStyle | 0.1083 | 0.2007 | 0.0923 | 0.485 | 0.015 |
| No consistency loss | 0.0557 | **0.1476** | **0.0919** | 0.513 | **0.013** |

**Key findings:**
- **DANN is essential** — removing it worsens ITW EER by 48% (0.1902 → 0.2808)
- **Multi-scale is essential** — single scale worsens ITW EER by 55% (0.1902 → 0.2952)
- **Consistency loss is counterproductive** — removing it improves both FoR and ITW; disc_acc closer to 0.5 (0.013 vs 0.087 deviation)
- **MixStyle helps marginally** — 5% ITW EER improvement

`disc_acc` = domain discriminator accuracy (target: 0.5 = discriminator at chance = domain-invariant features). Lower deviation from 0.5 indicates more effective DANN training.

---

## Project Structure

```
condetection-dann/
├── configs/default.yaml          # All hyperparameters
├── src/
│   ├── config.py                 # Dataclass config loader
│   ├── data/
│   │   ├── datasets.py           # FastAudioDataset, build_splits, make_loaders
│   │   ├── spectrograms.py       # Multi-resolution log-mel extraction (GPU)
│   │   └── augment.py            # SpecAugment, MixStyle, noise, time-stretch
│   ├── models/
│   │   ├── condetection.py       # ConDetection-DANN (main model)
│   │   ├── components.py         # GradReverse, ConformerBlock, CrossScaleAttentionFusion
│   │   ├── aasist.py             # AASIST baseline
│   │   ├── lcnn.py               # LCNN baseline
│   │   └── factory.py            # get_model(name, config)
│   ├── training/
│   │   ├── trainer.py            # train_one_epoch, evaluate, kfold_calibrate_threshold
│   │   ├── losses.py             # FocalBCE, consistency loss, DANN loss
│   │   ├── scheduler.py          # Cosine warmup scheduler
│   │   ├── ablation.py           # Ablation study runner (5 configs)
│   │   └── cv_trainer.py         # K-fold cross-validation
│   ├── evaluation/
│   │   ├── metrics.py            # EER, MinDCF, AUC, F1, bootstrap CI
│   │   ├── statistical.py        # McNemar, DeLong's test
│   │   └── comparative.py        # SOTA comparison pipeline
│   └── visualization/
│       ├── figures.py            # Training curves, ROC, confusion matrices
│       ├── gradcam.py            # Grad-CAM saliency maps
│       └── tsne.py               # t-SNE domain visualization
├── scripts/
│   ├── train.py                  # Main training entrypoint
│   ├── evaluate.py               # Standalone evaluation
│   ├── run_ablation.py           # Ablation study runner
│   ├── run_comparison.py         # SOTA comparative study
│   ├── run_cv.py                 # K-fold CV runner
│   ├── run_all.py                # End-to-end pipeline
│   └── generate_figures.py       # Publication figures
└── tests/                        # pytest test suite (6 files)
```

---

## Setup

```bash
# Clone and create environment
git clone https://github.com/Rishiatweb/Audio_Deepfake_Detection.git
cd Audio_Deepfake_Detection

# Install dependencies
pip install torch torchaudio librosa scikit-learn matplotlib seaborn pandas pyyaml scipy statsmodels
pip install -e .
```

### Dataset Paths

Edit `configs/default.yaml`:
```yaml
paths:
  for_base: /path/to/for-dataset      # FoR dataset root
  itw_root: /path/to/in-the-wild      # ITW dataset root
```

Or set environment variables:
```bash
export FOR_BASE=/path/to/for-dataset
export ITW_ROOT=/path/to/in-the-wild
```

---

## Usage

### Train ConDetection-DANN
```bash
python scripts/train.py --config configs/default.yaml
```

### Run SOTA Comparison
```bash
python scripts/run_comparison.py --config configs/default.yaml \
  --condetection-ckpt results/checkpoints/model_best.pt
```

### Run Ablation Study
```bash
python scripts/run_ablation.py --config configs/default.yaml --patience 3
```

### Run Full Pipeline
```bash
python scripts/run_all.py --config configs/default.yaml
```

### Generate Figures
```bash
python scripts/generate_figures.py --config configs/default.yaml \
  --ckpt results/checkpoints/model_best.pt
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 128 | Conformer hidden dimension |
| `n_layers` | 2 | Conformer blocks |
| `n_heads` | 4 | Attention heads |
| `batch_size` | 32 | Training batch size |
| `lr` | 3e-4 | Peak learning rate |
| `epochs` | 20 | Max training epochs |
| `patience` | 8 | Early stopping patience |
| `focal_gamma` | 1.5 | Focal loss gamma |
| `focal_alpha` | 0.60 | Focal loss alpha |
| `mixstyle_p` | 0.5 | MixStyle probability |
| `lambda_c` | 0.1 | Consistency loss weight |
| `dann.lambda_max` | 0.3 | Max DANN gradient reversal weight |
| `dann.warmup_epochs` | 4 | Epochs before DANN activates |

---

## Datasets

| Dataset | Role | Files | Source |
|---------|------|-------|--------|
| Fake or Real (FoR) | Train / Val / Test | 169,754 | [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) |
| In-the-Wild (ITW) | Out-of-domain test | 31,779 | [Kaggle](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake) |

---

## References

- Müller et al. (2022). *Does Audio Deepfake Detection Generalize?* — cross-domain generalization benchmark
- Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks* — DANN framework
- Gulati et al. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*
- Zhou et al. (2021). *MixStyle* — domain generalization via feature statistics mixing
- Park et al. (2019). *SpecAugment* — frequency and time masking augmentation
