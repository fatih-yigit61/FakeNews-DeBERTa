"""
FAZ C — Head Ablation Study (Minimum Viable)
Trains Only-Fake-Head model vs Full Model to prove multi-task learning value.

REQUIRES RETRAINING — uses extra Colab session.
Compares: Full Model (3 heads) vs Only Fake Head (1 head)

Usage (Colab):
    !python scripts/head_ablation.py
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    _dir = os.path.dirname(_dir)
    if os.path.isdir(os.path.join(_dir, "configs")):
        break
sys.path.insert(0, _dir)
os.chdir(_dir)

import json
import shutil
import numpy as np
import torch
from pathlib import Path

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer, run_full_pipeline

PLOT_DIR = Path("outputs/model1/plots")
ABLATION_DIR = Path("outputs/model1/ablation")


def train_only_fake(cfg):
    """Train with only the fake head active (manipulation & sentiment disabled)."""
    # Modify config to disable other heads
    cfg.lambda_manipulation = 0.0
    cfg.lambda_sentiment = 0.0
    cfg.lambda_fake = 1.0

    # Change output dir to avoid overwriting main model
    ablation_out = str(ABLATION_DIR / "only_fake")
    cfg.output_dir = ablation_out
    Path(ablation_out).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  ABLATION: Training Only-Fake-Head Model")
    print(f"  lambda_fake={cfg.lambda_fake}, lambda_manip={cfg.lambda_manipulation}, lambda_sent={cfg.lambda_sentiment}")
    print(f"  Output: {ablation_out}")
    print("=" * 62)

    trainer = Model1ExpertTrainer(cfg)
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.load_best_model()
    results = trainer.evaluate_all()
    return results


def load_full_model_results():
    """Load the full model's test results."""
    path = Path("outputs/model1/test_results.json")
    if not path.exists():
        print(f"[ERROR] Full model results not found at {path}")
        print("  Run full training first: python main.py --train")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 62)
    print("  HEAD ABLATION STUDY")
    print("  Full Model (3 heads) vs Only Fake Head (1 head)")
    print("=" * 62)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load full model results
    full_results = load_full_model_results()
    if full_results is None:
        return

    # Train only-fake model
    cfg = TrainerConfig()
    ablation_results = train_only_fake(cfg)

    # Compare
    print("\n" + "=" * 80)
    print(f"  {'Dataset':<20} {'Metric':<12} {'Full Model':>12} {'Only Fake':>12} {'Delta':>10}")
    print("  " + "-" * 68)

    comparisons = []
    for ds_name in ["gossipcop_test", "politifact_test", "liar_test"]:
        full_m = full_results.get(ds_name, {})
        abl_m = ablation_results.get(ds_name, {})

        full_acc = full_m.get("fake_acc", 0)
        abl_acc = abl_m.get("fake_acc", 0)
        delta = abl_acc - full_acc

        print(f"  {ds_name:<20} {'FakeAcc':<12} {full_acc:>12.4f} {abl_acc:>12.4f} {delta:>+10.4f}")
        comparisons.append({"dataset": ds_name, "full": full_acc, "ablated": abl_acc})

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = [c["dataset"] for c in comparisons]
    x = np.arange(len(datasets))
    w = 0.35

    ax.bar(x - w / 2, [c["full"] for c in comparisons], w,
           label="Full Model (3 heads)", color="steelblue")
    ax.bar(x + w / 2, [c["ablated"] for c in comparisons], w,
           label="Only Fake Head", color="coral")

    ax.set_ylabel("Fake/Real Accuracy")
    ax.set_title("Head Ablation Study: Multi-Task vs Single-Task")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_path = PLOT_DIR / "head_ablation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved → {save_path}")

    # Save results
    ablation_json = ABLATION_DIR / "comparison.json"
    with open(ablation_json, "w") as f:
        json.dump({"full_model": full_results, "only_fake": ablation_results}, f, indent=2)
    print(f"  [json] Saved → {ablation_json}")

    print(f"\n{'=' * 62}")
    print("  INTERPRETATION:")
    print("  Full > Only Fake → multi-task learning enriches shared encoder")
    print("  Full ≈ Only Fake → heads operate independently (still valid)")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
