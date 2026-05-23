"""Full experiment pipeline: train → compare → ablate → CV → figures."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
PYTHON = sys.executable


def run_step(name: str, cmd: list[str], skip: bool = False) -> bool:
    """Run one pipeline step. Returns True on success."""
    if skip:
        print(f"\n[SKIP] {name}")
        return True

    print(f"\n{'=' * 65}")
    print(f"[START] {name}")
    print("=" * 65)
    t0 = time.perf_counter()

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)

    elapsed = time.perf_counter() - t0
    mins, secs = divmod(int(elapsed), 60)
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {name} — {mins}m {secs}s")

    if result.returncode != 0:
        print(f"  Step exited with code {result.returncode}. Aborting pipeline.")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Run full ConDetection-DANN experiment pipeline")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--skip-comparison", action="store_true", help="Skip SOTA comparative study")
    p.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    p.add_argument("--skip-cv", action="store_true", help="Skip k-fold CV (slow: 5×10 epochs)")
    p.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    p.add_argument("--include-sklearn", action="store_true",
                   help="Include LR + RF baselines in comparative study")
    p.add_argument("--tsne", action="store_true", help="Generate t-SNE figures (very slow)")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--model", default="condetection",
                   help="Model for CV (condetection/aasist/lcnn/rawnet2)")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from existing checkpoint if present")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline_t0 = time.perf_counter()

    cfg = args.config
    ckpt = "results/checkpoints/model_best.pt"
    history = "results/training_history.csv"
    comp_out = "results/tables/comparative_results.csv"
    abl_out = "results/tables/ablation_results.csv"
    cv_out = f"results/tables/cv_results_{args.model}.csv"

    print("=" * 65)
    print("  ConDetection-DANN — Full Experiment Pipeline")
    print("=" * 65)
    skips = []
    if args.skip_comparison:
        skips.append("comparison")
    if args.skip_ablation:
        skips.append("ablation")
    if args.skip_cv:
        skips.append("CV")
    if args.skip_figures:
        skips.append("figures")
    if skips:
        print(f"  Skipping: {', '.join(skips)}")
    print()

    # ── Step 1: Train ──────────────────────────────────────────────────────────
    train_cmd = [PYTHON, "scripts/train.py", "--config", cfg]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.resume:
        train_cmd += ["--resume", str(Path(ROOT / "results/checkpoints/checkpoint_best.pth"))]
    if not run_step("Train ConDetection-DANN", train_cmd):
        sys.exit(1)

    # ── Step 2: SOTA Comparative Study ────────────────────────────────────────
    comp_cmd = [
        PYTHON, "scripts/run_comparison.py",
        "--config", cfg,
        "--condetection-ckpt", ckpt,
        "--output", comp_out,
    ]
    if args.include_sklearn:
        comp_cmd.append("--include-sklearn")
    if not run_step("SOTA Comparative Study", comp_cmd, skip=args.skip_comparison):
        sys.exit(1)

    # ── Step 3: Ablation Study ────────────────────────────────────────────────
    abl_cmd = [
        PYTHON, "scripts/run_ablation.py",
        "--config", cfg,
        "--output", abl_out,
    ]
    if not run_step("Ablation Study", abl_cmd, skip=args.skip_ablation):
        sys.exit(1)

    # ── Step 4: K-fold Cross-Validation ───────────────────────────────────────
    cv_cmd = [
        PYTHON, "scripts/run_cv.py",
        "--config", cfg,
        "--model", args.model,
        "--output", cv_out,
    ]
    if not run_step("K-fold Cross-Validation", cv_cmd, skip=args.skip_cv):
        sys.exit(1)

    # ── Step 5: Generate Figures ──────────────────────────────────────────────
    fig_cmd = [
        PYTHON, "scripts/generate_figures.py",
        "--config", cfg,
        "--ckpt", ckpt,
        "--history", history,
    ]
    if Path(ROOT / comp_out).exists():
        fig_cmd += ["--comparison", comp_out]
    if Path(ROOT / abl_out).exists():
        fig_cmd += ["--ablation", abl_out]
    if args.tsne:
        fig_cmd.append("--tsne")
    if not run_step("Generate Publication Figures", fig_cmd, skip=args.skip_figures):
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_t0
    hours, rem = divmod(int(total), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n{'=' * 65}")
    print(f"  PIPELINE COMPLETE — {hours}h {mins}m {secs}s")
    print(f"  Checkpoint : {ckpt}")
    print("  Figures    : results/figures/")
    print("  Tables     : results/tables/")
    print("=" * 65)


if __name__ == "__main__":
    main()
