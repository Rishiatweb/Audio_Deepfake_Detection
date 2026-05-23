# ConDetection-DANN

Multi-scale Conformer with Domain-Adversarial Neural Network (DANN) for audio deepfake detection.  
Publication target: Scopus-indexed Indian conference, 2026.

## Architecture

- **Multi-scale mel encoders**: fine (n_fft=400), mid (1024), coarse (2048)
- **Shared Conformer blocks** with pooling between layers
- **Cross-Scale Attention Fusion**: multi-head attention across 3 scales
- **DANN with Gradient Reversal Layer**: domain-invariant feature learning
- **Consistency loss**: enforces agreement across scale embeddings
- ~1.1M trainable parameters

## Datasets

| Dataset | Size | Download |
|---------|------|----------|
| FoR (Fake-or-Real) | ~17 GB, 4 subsets | Kaggle: `mohammedabdeldayem/the-fake-or-real-dataset` |
| In-the-Wild (ITW) | ~8 GB | HuggingFace: `mozilla-foundation/common_voice_...` / `muller91/in-the-wild` |

After downloading, set paths in `configs/default.yaml`:
```yaml
paths:
  for_base: /path/to/for-dataset      # contains for-original/, for-norm/, etc.
  itw_root: /path/to/in-the-wild      # contains real/ and fake/ subdirs
```

Or override via CLI:
```bash
python scripts/train.py --for-base /path/to/for-dataset --itw-root /path/to/in-the-wild
```

## Setup

Requires Python 3.12, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Rishiatweb/Audio_Deepfake_Detection
cd Audio_Deepfake_Detection
uv sync
```

## Running Experiments

### 1. Train ConDetection-DANN (main model)
```bash
python scripts/train.py --config configs/default.yaml
```

### 2. Evaluate a checkpoint
```bash
python scripts/evaluate.py --ckpt results/checkpoints/model_best.pt
```

### 3. SOTA comparative study (trains LCNN, AASIST, RawNet2)
```bash
python scripts/run_comparison.py \
    --condetection-ckpt results/checkpoints/model_best.pt

# Include sklearn baselines (LR + RF):
python scripts/run_comparison.py --include-sklearn \
    --condetection-ckpt results/checkpoints/model_best.pt
```

### 4. Ablation study (5 variants)
```bash
python scripts/run_ablation.py --config configs/default.yaml
```

### 5. K-fold cross-validation (5 folds)
```bash
python scripts/run_cv.py --config configs/default.yaml --model condetection
```

### 6. Generate publication figures
```bash
python scripts/generate_figures.py \
    --ckpt results/checkpoints/model_best.pt \
    --history results/training_history.csv \
    --comparison results/tables/comparative_results.csv \
    --ablation results/tables/ablation_results.csv
```

Figures saved to `results/figures/`.

## Code Quality

```bash
ruff check src/ scripts/ tests/      # lint
ruff format src/ scripts/ tests/     # format
pylint src/                          # static analysis
pytest tests/ -v                     # test suite
```

## Project Structure

```
condetection-dann/
├── configs/default.yaml         # All hyperparameters
├── src/
│   ├── config.py                # Config dataclasses + YAML loader
│   ├── data/                    # datasets, spectrograms, augmentations
│   ├── models/                  # ConDetection, AASIST, RawNet2, LCNN, factory
│   ├── training/                # trainer, losses, scheduler, ablation, cv_trainer
│   ├── evaluation/              # metrics, statistical tests, comparative study
│   └── visualization/           # figures, Grad-CAM, t-SNE
├── scripts/
│   ├── train.py                 # Main training entrypoint
│   ├── evaluate.py              # Standalone evaluation
│   ├── run_comparison.py        # SOTA comparison
│   ├── run_ablation.py          # Ablation study
│   ├── run_cv.py                # K-fold cross-validation
│   └── generate_figures.py      # Publication figures
├── tests/                       # pytest test suite (~1000 lines)
└── notebooks/paper_figures.ipynb
```

## Key Results (after full training)

Results saved to `results/tables/comparative_results.csv` after running `run_comparison.py`.

| Metric | FoR Test | In-the-Wild |
|--------|----------|-------------|
| EER (lower better) | — | — |
| AUC (higher better) | — | — |
| MinDCF | — | — |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{condetection2026,
  title     = {ConDetection-DANN: Domain-Adversarial Multi-Scale Conformer for Audio Deepfake Detection},
  author    = {TODO},
  booktitle = {TODO (Scopus-indexed Indian conference)},
  year      = {2026},
}
```
