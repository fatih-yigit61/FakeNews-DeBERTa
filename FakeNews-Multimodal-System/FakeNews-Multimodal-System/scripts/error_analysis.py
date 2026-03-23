"""
FAZ B4 — Error Analysis
Examines misclassified samples to find patterns in model errors.
Answers: "What kinds of news does the model get wrong?"

Usage (Colab, after training):
    !python scripts/error_analysis.py
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
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
from pathlib import Path
from collections import Counter

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer

PLOT_DIR = Path("outputs/model1/plots")


def collect_predictions(trainer, loader):
    """Collect all predictions with texts for error analysis."""
    trainer.model.eval()
    device = trainer.device

    all_items = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            style = batch["style_feats"].to(device)

            with autocast("cuda", enabled=trainer.cfg.fp16):
                preds = trainer.model(input_ids, attn_mask, style)

            fake_softmax = torch.softmax(preds["fake_logits"], dim=-1).cpu().numpy()
            fake_label = batch["fake_label"].numpy()

            for j in range(len(fake_label)):
                if fake_label[j] == -1:
                    continue

                # Decode text from input_ids
                token_ids = input_ids[j].cpu().tolist()
                text = trainer.tokenizer.decode(token_ids, skip_special_tokens=True)

                all_items.append({
                    "text": text[:200],  # first 200 chars
                    "true_label": int(fake_label[j]),
                    "pred_label": int(fake_softmax[j].argmax()),
                    "fake_prob": float(fake_softmax[j, 1]),
                    "confidence": float(fake_softmax[j].max()),
                    "text_len": len(text.split()),
                })

    return all_items


def main():
    print("=" * 62)
    print("  ERROR ANALYSIS")
    print("=" * 62)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig()
    trainer = Model1ExpertTrainer(cfg)
    trainer.load_data()
    trainer.build_model()
    trainer.load_best_model()

    for name, loader in trainer.test_loaders.items():
        items = collect_predictions(trainer, loader)
        if not items:
            continue

        correct = [it for it in items if it["true_label"] == it["pred_label"]]
        errors = [it for it in items if it["true_label"] != it["pred_label"]]

        if not errors:
            print(f"\n  {name}: 0 errors out of {len(items)} — perfect!")
            continue

        # Error breakdown
        fp = [e for e in errors if e["true_label"] == 0 and e["pred_label"] == 1]  # Real→Fake
        fn = [e for e in errors if e["true_label"] == 1 and e["pred_label"] == 0]  # Fake→Real

        print(f"\n{'─' * 62}")
        print(f"  {name}: {len(errors)} errors / {len(items)} total ({len(errors)/len(items):.1%} error rate)")
        print(f"  False Positives (Real→Fake): {len(fp)}")
        print(f"  False Negatives (Fake→Real): {len(fn)}")

        # Text length analysis
        error_lengths = [e["text_len"] for e in errors]
        correct_lengths = [c["text_len"] for c in correct]
        print(f"\n  Avg text length — Errors: {np.mean(error_lengths):.0f} words, "
              f"Correct: {np.mean(correct_lengths):.0f} words")

        # Confidence analysis
        error_confs = [e["confidence"] for e in errors]
        correct_confs = [c["confidence"] for c in correct]
        print(f"  Avg confidence — Errors: {np.mean(error_confs):.3f}, "
              f"Correct: {np.mean(correct_confs):.3f}")

        # Show some example errors
        print(f"\n  Top 5 False Positives (Real articles predicted as Fake):")
        for e in sorted(fp, key=lambda x: x["fake_prob"], reverse=True)[:5]:
            print(f"    [{e['fake_prob']:.1%} fake] {e['text'][:100]}...")

        print(f"\n  Top 5 False Negatives (Fake articles predicted as Real):")
        for e in sorted(fn, key=lambda x: x["fake_prob"])[:5]:
            print(f"    [{e['fake_prob']:.1%} fake] {e['text'][:100]}...")

        # Plot: Error confidence distribution
        if len(errors) >= 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Confidence distribution
            axes[0].hist([e["confidence"] for e in errors], bins=20, alpha=0.7,
                        color="coral", edgecolor="darkred", label="Errors")
            axes[0].hist([c["confidence"] for c in correct[:500]], bins=20, alpha=0.5,
                        color="steelblue", edgecolor="navy", label="Correct (sample)")
            axes[0].set_xlabel("Model Confidence")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"{name}: Confidence Distribution")
            axes[0].legend()

            # Text length distribution
            axes[1].hist([e["text_len"] for e in errors], bins=20, alpha=0.7,
                        color="coral", edgecolor="darkred", label="Errors")
            axes[1].hist([c["text_len"] for c in correct[:500]], bins=20, alpha=0.5,
                        color="steelblue", edgecolor="navy", label="Correct (sample)")
            axes[1].set_xlabel("Text Length (words)")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"{name}: Text Length Distribution")
            axes[1].legend()

            # Error type pie chart
            labels_pie = [f"False Positive\n(Real→Fake)\n{len(fp)}",
                         f"False Negative\n(Fake→Real)\n{len(fn)}"]
            sizes = [len(fp), len(fn)]
            if sum(sizes) > 0:
                axes[2].pie(sizes, labels=labels_pie, colors=["#ff6b6b", "#4ecdc4"],
                           autopct="%1.0f%%", startangle=90)
                axes[2].set_title(f"{name}: Error Types")

            fig.suptitle(f"Error Analysis — {name}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            save_path = PLOT_DIR / f"error_analysis_{name}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  [plot] Saved → {save_path}")

    # Save all errors to JSON
    print(f"\n{'=' * 62}")
    print("  Error analysis complete. Check plots in outputs/model1/plots/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
