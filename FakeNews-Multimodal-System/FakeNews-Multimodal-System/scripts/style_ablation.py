"""
FAZ B1 — Style Feature Ablation Study
Compares model performance WITH vs WITHOUT stylometric features.
No retraining needed — zeroes out style_feats at test time.

Thesis interpretation (both outcomes are publishable):
  - If performance DROPS  → model learned useful stylistic patterns
  - If performance STAYS  → model learned semantics, not dependent on style

Usage (Colab, after training):
    !python scripts/style_ablation.py
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    _dir = os.path.dirname(_dir)
    if os.path.isdir(os.path.join(_dir, "configs")):
        break
sys.path.insert(0, _dir)
os.chdir(_dir)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer

PLOT_DIR = Path("outputs/model1/plots")


def evaluate_with_style_mode(trainer, loader, zero_style=False, manip_thr=0.50):
    """Evaluate a loader, optionally zeroing style_feats."""
    trainer.model.eval()
    device = trainer.device

    manip_probs, manip_labels = [], []
    fake_preds, fake_labels = [], []
    sent_preds, sent_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            style = batch["style_feats"].to(device)

            # ABLATION: zero out style features
            if zero_style:
                style = torch.zeros_like(style)

            with autocast("cuda", enabled=trainer.cfg.fp16):
                predictions = trainer.model(input_ids, attn_mask, style)

            # Manipulation
            ml = batch["manipulation_label"].numpy()
            mm = ml != -1
            if mm.any():
                probs = torch.sigmoid(predictions["manipulation_logits"]).cpu().numpy()
                manip_probs.extend(probs[mm].tolist())
                manip_labels.extend(ml[mm].astype(int).tolist())

            # Fake
            fl = batch["fake_label"].numpy()
            fm = fl != -1
            if fm.any():
                fp = torch.softmax(predictions["fake_logits"], dim=-1).argmax(dim=-1).cpu().numpy()
                fake_preds.extend(fp[fm].tolist())
                fake_labels.extend(fl[fm].tolist())

            # Sentiment
            sl = batch["sentiment_label"].numpy()
            sm = sl != -1
            if sm.any():
                sp = predictions["sentiment_logits"].argmax(dim=-1).cpu().numpy()
                sent_preds.extend(sp[sm].tolist())
                sent_labels.extend(sl[sm].tolist())

    results = {}
    if fake_labels:
        results["fake_acc"] = accuracy_score(fake_labels, fake_preds)
    if manip_labels:
        mp = (np.array(manip_probs) >= manip_thr).astype(int)
        results["manip_f1"] = f1_score(manip_labels, mp, zero_division=0)
    if sent_labels:
        results["sent_acc"] = accuracy_score(sent_labels, sent_preds)
    return results


def main():
    print("=" * 62)
    print("  STYLE FEATURE ABLATION STUDY")
    print("=" * 62)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig()
    trainer = Model1ExpertTrainer(cfg)
    trainer.load_data()
    trainer.build_model()
    trainer.load_best_model()

    manip_thr = trainer._manip_threshold
    print(f"  Using manipulation threshold: {manip_thr:.2f}")

    # Collect results for both modes
    all_results = {}

    test_sets = {"val": trainer.val_loader}
    test_sets.update(trainer.test_loaders)

    for name, loader in test_sets.items():
        print(f"\n  Evaluating: {name}")
        normal = evaluate_with_style_mode(trainer, loader, zero_style=False, manip_thr=manip_thr)
        ablated = evaluate_with_style_mode(trainer, loader, zero_style=True, manip_thr=manip_thr)
        all_results[name] = {"normal": normal, "ablated": ablated}

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"  {'Dataset':<20} {'Metric':<12} {'Normal':>10} {'Style=0':>10} {'Delta':>10} {'Impact':>10}")
    print("  " + "-" * 72)

    for name, data in all_results.items():
        for metric in ["fake_acc", "manip_f1", "sent_acc"]:
            if metric in data["normal"]:
                n = data["normal"][metric]
                a = data["ablated"][metric]
                delta = a - n
                impact = "DROPS" if delta < -0.01 else ("IMPROVES" if delta > 0.01 else "STABLE")
                label = {"fake_acc": "FakeAcc", "manip_f1": "ManipF1", "sent_acc": "SentAcc"}[metric]
                print(f"  {name:<20} {label:<12} {n:>10.4f} {a:>10.4f} {delta:>+10.4f} {impact:>10}")

    # Bar chart for thesis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("fake_acc", "Fake/Real Accuracy", axes[0]),
               ("manip_f1", "Manipulation F1", axes[1]),
               ("sent_acc", "Sentiment Accuracy", axes[2])]

    for metric, title, ax in metrics:
        datasets = []
        normal_vals = []
        ablated_vals = []
        for name, data in all_results.items():
            if metric in data["normal"]:
                datasets.append(name)
                normal_vals.append(data["normal"][metric])
                ablated_vals.append(data["ablated"][metric])

        if not datasets:
            continue

        x = np.arange(len(datasets))
        w = 0.35
        ax.bar(x - w / 2, normal_vals, w, label="With Style Features", color="steelblue")
        ax.bar(x + w / 2, ablated_vals, w, label="Without Style Features", color="coral")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    fig.suptitle("Style Feature Ablation Study", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = PLOT_DIR / "style_ablation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved → {save_path}")

    print(f"\n{'=' * 62}")
    print("  INTERPRETATION GUIDE:")
    print("  If scores DROP  → model uses stylistic patterns (not just word memorization)")
    print("  If scores STABLE → model learned semantics (style-independent)")
    print("  Both outcomes are valid thesis findings.")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
