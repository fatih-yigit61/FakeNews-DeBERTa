"""
Post-hoc threshold & calibration tuning — run AFTER training.
Uses validation set to find optimal thresholds, then evaluates on test sets.
No retraining needed.  Generates plots for thesis.

Usage (Colab):
    %cd /content/FakeNews-Multimodal-System/FakeNews-Multimodal-System
    !python scripts/threshold_tuning.py
"""
import sys, os
# Resolve project root: walk up from script dir until we find configs/
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pathlib import Path

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer

PLOT_DIR = Path("outputs/model1/plots")


def collect_raw_outputs(trainer, loader):
    """Run model on a DataLoader and collect raw logits/probs + labels."""
    trainer.model.eval()
    device = trainer.device

    manip_probs_all, manip_labels_all = [], []
    sent_logits_all, sent_labels_all = [], []
    fake_probs_all, fake_labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            style = batch["style_feats"].to(device)

            with autocast("cuda", enabled=trainer.cfg.fp16):
                predictions = trainer.model(input_ids, attn_mask, style)

            # Manipulation
            manip_label = batch["manipulation_label"].numpy()
            manip_mask = manip_label != -1
            if manip_mask.any():
                probs = torch.sigmoid(predictions["manipulation_logits"]).cpu().numpy()
                manip_probs_all.extend(probs[manip_mask].tolist())
                manip_labels_all.extend(manip_label[manip_mask].astype(int).tolist())

            # Sentiment (raw logits for temperature scaling)
            sent_label = batch["sentiment_label"].numpy()
            sent_mask = sent_label != -1
            if sent_mask.any():
                logits = predictions["sentiment_logits"].cpu().numpy()
                sent_logits_all.extend(logits[sent_mask].tolist())
                sent_labels_all.extend(sent_label[sent_mask].astype(int).tolist())

            # Fake
            fake_label = batch["fake_label"].numpy()
            fake_mask = fake_label != -1
            if fake_mask.any():
                fake_softmax = torch.softmax(predictions["fake_logits"], dim=-1).cpu().numpy()
                fake_probs_all.extend(fake_softmax[fake_mask, 1].tolist())
                fake_labels_all.extend(fake_label[fake_mask].astype(int).tolist())

    return {
        "manip_probs": np.array(manip_probs_all),
        "manip_labels": np.array(manip_labels_all),
        "sent_logits": np.array(sent_logits_all),
        "sent_labels": np.array(sent_labels_all),
        "fake_probs": np.array(fake_probs_all),
        "fake_labels": np.array(fake_labels_all),
    }


# ═══════════════════════════════════════════════════════════════
#  MANIPULATION THRESHOLD
# ═══════════════════════════════════════════════════════════════

def sweep_manipulation_threshold(probs, labels, start=0.20, end=0.80, step=0.01):
    """Sweep manipulation threshold and return ALL results (for plotting) + sorted by F1."""
    results = []
    for thr in np.arange(start, end + step, step):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        results.append({"threshold": round(thr, 2), "F1": round(f1, 4),
                        "Precision": round(p, 4), "Recall": round(r, 4)})
    return results


def plot_threshold_vs_f1(results, best_thr, save_path):
    """Plot threshold vs F1/Precision/Recall — thesis figure."""
    thresholds = [r["threshold"] for r in results]
    f1s = [r["F1"] for r in results]
    precs = [r["Precision"] for r in results]
    recs = [r["Recall"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, "b-", linewidth=2, label="F1 Score")
    ax.plot(thresholds, precs, "g--", linewidth=1.5, label="Precision")
    ax.plot(thresholds, recs, "r--", linewidth=1.5, label="Recall")

    # Mark best threshold
    best_f1 = max(f1s)
    ax.axvline(x=best_thr, color="blue", linestyle=":", alpha=0.7)
    ax.scatter([best_thr], [best_f1], color="blue", s=100, zorder=5,
               label=f"Best: thr={best_thr:.2f}, F1={best_f1:.4f}")

    # Mark original threshold (0.49)
    idx_049 = next((i for i, r in enumerate(results) if r["threshold"] == 0.49), None)
    if idx_049 is not None:
        ax.scatter([0.49], [f1s[idx_049]], color="red", s=80, marker="x", zorder=5,
                   label=f"Original: thr=0.49, F1={f1s[idx_049]:.4f}")

    # Target line
    ax.axhline(y=0.65, color="orange", linestyle="--", alpha=0.5, label="Target F1=0.65")

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Manipulation Detection: Threshold Calibration", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.20, 0.80)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved threshold vs F1 → {save_path}")


# ═══════════════════════════════════════════════════════════════
#  SENTIMENT CALIBRATION
# ═══════════════════════════════════════════════════════════════

def sweep_sentiment_calibration(logits, labels):
    """Try temperature scaling and neutral bias to improve sentiment accuracy."""
    results = []
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_np = np.array(labels)

    # Baseline (no calibration)
    baseline_preds = logits_t.argmax(dim=-1).numpy()
    baseline_acc = accuracy_score(labels_np, baseline_preds)
    results.append({"temp": 1.0, "neutral_bias": 0.0, "acc": round(baseline_acc, 4), "tag": "baseline"})

    # Temperature scaling
    for temp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        scaled = logits_t / temp
        preds = scaled.argmax(dim=-1).numpy()
        acc = accuracy_score(labels_np, preds)
        results.append({"temp": temp, "neutral_bias": 0.0, "acc": round(acc, 4), "tag": f"temp={temp}"})

    # Temperature + neutral bias (class 1 = neutral)
    for temp in [0.5, 0.7, 1.0, 1.5, 2.0]:
        for bias in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            scaled = logits_t / temp
            scaled[:, 1] += bias
            preds = scaled.argmax(dim=-1).numpy()
            acc = accuracy_score(labels_np, preds)
            results.append({"temp": temp, "neutral_bias": bias, "acc": round(acc, 4),
                           "tag": f"temp={temp}_bias={bias}"})

    # Per-class bias adjustments
    for temp in [0.7, 1.0, 1.5]:
        for neg_bias in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            for pos_bias in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
                if neg_bias == 0.0 and pos_bias == 0.0:
                    continue
                scaled = logits_t / temp
                scaled[:, 0] += neg_bias
                scaled[:, 2] += pos_bias
                preds = scaled.argmax(dim=-1).numpy()
                acc = accuracy_score(labels_np, preds)
                results.append({"temp": temp, "neutral_bias": 0.0, "acc": round(acc, 4),
                               "tag": f"temp={temp}_neg={neg_bias}_pos={pos_bias}"})

    results.sort(key=lambda x: x["acc"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════
#  RELIABILITY DIAGRAM (Confidence Calibration)
# ═══════════════════════════════════════════════════════════════

def plot_reliability_diagram(probs, labels, title, save_path, n_bins=10):
    """Reliability diagram + ECE — shows if model confidence is calibrated."""
    probs = np.array(probs)
    labels = np.array(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(probs[mask].mean())
        bin_counts.append(mask.sum())

    # ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(abs(a - c) * n / total for a, c, n in zip(bin_accs, bin_confs, bin_counts) if n > 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Reliability diagram
    bar_width = 1.0 / n_bins
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]

    ax1.bar(bin_centers, bin_accs, width=bar_width * 0.8, alpha=0.7, color="steelblue",
            edgecolor="navy", label="Model accuracy")
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right: Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=bar_width * 0.8, alpha=0.7, color="coral",
            edgecolor="darkred")
    ax2.set_xlabel("Predicted Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved reliability diagram → {save_path}")
    print(f"  [metric] ECE = {ece:.4f}")
    return ece


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  POST-HOC THRESHOLD & CALIBRATION TUNING")
    print("=" * 62)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load trainer and data
    cfg = TrainerConfig()
    trainer = Model1ExpertTrainer(cfg)
    trainer.load_data()

    # Load best model checkpoint
    ckpt_path = Path(cfg.output_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint at {ckpt_path}")
        print("  Run training first, or copy checkpoint to this path.")
        return
    trainer.load_best_model()
    print(f"\nLoaded model from {ckpt_path}")

    # ─── Validation set: find optimal thresholds ─────────────────
    print("\n" + "─" * 62)
    print("  STEP 1: Collecting raw outputs on VALIDATION set")
    print("─" * 62)
    val_data = collect_raw_outputs(trainer, trainer.val_loader)

    # ── Manipulation threshold sweep ─────────────────────────────
    print(f"\n  Manipulation samples: {len(val_data['manip_labels'])}")
    best_manip_thr = 0.50
    if len(val_data["manip_labels"]) > 0:
        manip_results = sweep_manipulation_threshold(val_data["manip_probs"], val_data["manip_labels"])
        # Sort by F1 for display
        sorted_by_f1 = sorted(manip_results, key=lambda x: x["F1"], reverse=True)
        print("\n  TOP 10 Manipulation Thresholds (val):")
        print(f"  {'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        for r in sorted_by_f1[:10]:
            marker = " <-- BEST" if r == sorted_by_f1[0] else ""
            print(f"  {r['threshold']:>10.2f} {r['F1']:>8.4f} {r['Precision']:>10.4f} {r['Recall']:>8.4f}{marker}")
        best_manip_thr = sorted_by_f1[0]["threshold"]

        # Plot threshold vs F1 (thesis figure)
        plot_threshold_vs_f1(manip_results, best_manip_thr,
                            PLOT_DIR / "manipulation_threshold_calibration.png")
    else:
        print("  [warn] No manipulation samples in validation set")

    # ── Sentiment calibration sweep ──────────────────────────────
    print(f"\n  Sentiment samples: {len(val_data['sent_labels'])}")
    best_sent_config = {"temp": 1.0, "neutral_bias": 0.0, "tag": "baseline"}
    if len(val_data["sent_labels"]) > 0:
        sent_results = sweep_sentiment_calibration(val_data["sent_logits"], val_data["sent_labels"])
        print("\n  TOP 15 Sentiment Calibrations (val):")
        print(f"  {'Tag':>35} {'Accuracy':>10}")
        for r in sent_results[:15]:
            marker = " <-- BEST" if r == sent_results[0] else ""
            print(f"  {r['tag']:>35} {r['acc']:>10.4f}{marker}")
        best_sent_config = sent_results[0]
    else:
        print("  [warn] No sentiment samples in validation set")

    # ── Reliability diagrams (confidence calibration) ────────────
    print("\n" + "─" * 62)
    print("  STEP 2: Confidence Calibration Analysis")
    print("─" * 62)

    if len(val_data["fake_probs"]) > 0:
        # For fake detection: use max(P(real), P(fake)) as confidence
        fake_confs = np.maximum(val_data["fake_probs"], 1 - val_data["fake_probs"])
        fake_preds = (val_data["fake_probs"] >= 0.5).astype(int)
        fake_correct = (fake_preds == val_data["fake_labels"]).astype(int)
        plot_reliability_diagram(fake_confs, fake_correct,
                                "Fake/Real Detection — Confidence Calibration",
                                PLOT_DIR / "reliability_fake.png")

    if len(val_data["manip_probs"]) > 0:
        manip_confs = np.maximum(val_data["manip_probs"], 1 - val_data["manip_probs"])
        manip_preds = (val_data["manip_probs"] >= best_manip_thr).astype(int)
        manip_correct = (manip_preds == val_data["manip_labels"]).astype(int)
        plot_reliability_diagram(manip_confs, manip_correct,
                                "Manipulation Detection — Confidence Calibration",
                                PLOT_DIR / "reliability_manipulation.png")

    if len(val_data["sent_logits"]) > 0:
        sent_softmax = torch.softmax(torch.tensor(val_data["sent_logits"], dtype=torch.float32), dim=-1).numpy()
        sent_confs = sent_softmax.max(axis=-1)
        sent_preds = sent_softmax.argmax(axis=-1)
        sent_correct = (sent_preds == np.array(val_data["sent_labels"])).astype(int)
        plot_reliability_diagram(sent_confs, sent_correct,
                                "Sentiment Analysis — Confidence Calibration",
                                PLOT_DIR / "reliability_sentiment.png")

    # ─── Apply best thresholds to TEST sets ──────────────────────
    print("\n" + "─" * 62)
    print("  STEP 3: Applying best thresholds to TEST sets")
    print("─" * 62)
    print(f"  Best manipulation threshold: {best_manip_thr}")
    print(f"  Best sentiment config: {best_sent_config.get('tag', 'baseline')}")

    for name, loader in trainer.test_loaders.items():
        print(f"\n  +-- {name} {'--' * (25 - len(name))}")
        test_data = collect_raw_outputs(trainer, loader)

        # Manipulation with optimal threshold
        if len(test_data["manip_labels"]) > 0:
            preds = (test_data["manip_probs"] >= best_manip_thr).astype(int)
            f1 = f1_score(test_data["manip_labels"], preds, zero_division=0)
            p = precision_score(test_data["manip_labels"], preds, zero_division=0)
            r = recall_score(test_data["manip_labels"], preds, zero_division=0)

            # Also show with original threshold for comparison
            orig_preds = (test_data["manip_probs"] >= 0.49).astype(int)
            orig_f1 = f1_score(test_data["manip_labels"], orig_preds, zero_division=0)

            print(f"  |  Manipulation (thr={best_manip_thr:.2f}): F1={f1:.4f}  P={p:.4f}  R={r:.4f}")
            print(f"  |  Manipulation (thr=0.49 orig):  F1={orig_f1:.4f}")
            delta = f1 - orig_f1
            print(f"  |  Delta: {'+' if delta >= 0 else ''}{delta:.4f}")

        # Sentiment with optimal calibration
        if len(test_data["sent_labels"]) > 0:
            logits_t = torch.tensor(test_data["sent_logits"], dtype=torch.float32)
            labels_np = test_data["sent_labels"]

            # Original accuracy
            orig_preds = logits_t.argmax(dim=-1).numpy()
            orig_acc = accuracy_score(labels_np, orig_preds)

            # Calibrated accuracy
            temp = best_sent_config.get("temp", 1.0)
            scaled = logits_t / temp
            if "neutral_bias" in best_sent_config and best_sent_config["neutral_bias"] > 0:
                scaled[:, 1] += best_sent_config["neutral_bias"]
            tag = best_sent_config.get("tag", "")
            if "neg=" in tag and "pos=" in tag:
                parts = tag.split("_")
                for part in parts:
                    if part.startswith("neg="):
                        scaled[:, 0] += float(part[4:])
                    elif part.startswith("pos="):
                        scaled[:, 2] += float(part[4:])

            cal_preds = scaled.argmax(dim=-1).numpy()
            cal_acc = accuracy_score(labels_np, cal_preds)

            print(f"  |  Sentiment (calibrated): Acc={cal_acc:.4f}")
            print(f"  |  Sentiment (original):   Acc={orig_acc:.4f}")
            delta = cal_acc - orig_acc
            print(f"  |  Delta: {'+' if delta >= 0 else ''}{delta:.4f}")

        # Fake/Real
        if len(test_data["fake_labels"]) > 0:
            preds = (np.array(test_data["fake_probs"]) >= 0.5).astype(int)
            acc = accuracy_score(test_data["fake_labels"], preds)
            print(f"  |  Fake/Real: Acc={acc:.4f}")

        print(f"  +{'--' * 28}")

    # ─── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print(f"{'=' * 62}")
    print(f"  Optimal manipulation threshold : {best_manip_thr}")
    print(f"  Optimal sentiment calibration  : {best_sent_config.get('tag', 'baseline')}")
    print(f"\n  Plots saved to {PLOT_DIR}/:")
    print(f"    - manipulation_threshold_calibration.png  (thesis figure)")
    print(f"    - reliability_fake.png                    (confidence calibration)")
    print(f"    - reliability_manipulation.png            (confidence calibration)")
    print(f"    - reliability_sentiment.png               (confidence calibration)")
    print(f"\n  Next: apply these values to config.py if they improve test metrics")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
