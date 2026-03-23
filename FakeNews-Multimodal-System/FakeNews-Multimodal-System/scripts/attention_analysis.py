"""
FAZ B3 — Attention / Integrated Gradients Visualization
Shows which tokens the model focuses on when making predictions.

IMPORTANT THESIS DISCLAIMER:
  "Attention weights provide an indication of focus but do not
   necessarily reflect causal importance." (Jain & Wallace, 2019)

Usage (Colab, after training):
    !pip install captum -q
    !python scripts/attention_analysis.py
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
import matplotlib.colors as mcolors
from torch.amp import autocast
from pathlib import Path

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer
from src.features.stylometry import StylometricExtractor, StyleScaler

PLOT_DIR = Path("outputs/model1/plots")

# Sample texts for visualization
SAMPLE_TEXTS = [
    {
        "text": "The Federal Reserve announced a 0.25 percent interest rate cut after a two-day meeting, "
                "citing stable inflation expectations and moderate economic growth in the third quarter.",
        "expected": "real",
        "label": "Real news — neutral, factual reporting",
    },
    {
        "text": "SHOCKING EXPOSED!!! Government SECRETLY funding alien research!!! "
                "Scientists TERRIFIED as leaked documents reveal the TRUTH about Area 51!!! "
                "They tried to HIDE this from you!!!",
        "expected": "fake",
        "label": "Fake news — sensationalist, manipulative language",
    },
    {
        "text": "A new study published in Nature found that climate change has accelerated "
                "the melting of Arctic ice sheets by 30 percent over the past decade.",
        "expected": "real",
        "label": "Real news — scientific reporting with data",
    },
    {
        "text": "BREAKING BOMBSHELL!!! Celebrity doctor reveals the ONE food that CURES cancer!!! "
                "Big Pharma is FURIOUS and trying to BAN this miracle cure!!! Share before they DELETE this!!!",
        "expected": "fake",
        "label": "Fake news — health misinformation with urgency",
    },
]


def get_attention_weights(trainer, text):
    """Extract attention weights from the last layer of XLM-RoBERTa."""
    enc = trainer.tokenizer(
        text, max_length=trainer.cfg.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(trainer.device)
    attn_mask = enc["attention_mask"].to(trainer.device)

    # Switch to eager attention to support output_attentions
    encoder = trainer.model.encoder
    orig_impl = getattr(encoder.config, "_attn_implementation", None)
    encoder.config._attn_implementation = "eager"

    trainer.model.eval()
    with torch.no_grad():
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )

    # Restore original attention implementation
    if orig_impl is not None:
        encoder.config._attn_implementation = orig_impl

    # Last layer attention, average across heads: [1, num_heads, seq_len, seq_len]
    last_attn = outputs.attentions[-1]  # [1, 12, seq_len, seq_len]
    # Average across heads and take attention FROM [CLS] token (index 0)
    cls_attn = last_attn[0].mean(dim=0)[0]  # [seq_len]

    # Get actual tokens (not padding)
    tokens = trainer.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    actual_len = attn_mask[0].sum().item()

    return tokens[:actual_len], cls_attn[:actual_len].cpu().numpy()


def try_integrated_gradients(trainer, text):
    """Try Integrated Gradients via Captum (if installed)."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        print("  [info] Captum not installed, skipping Integrated Gradients.")
        print("         Install with: pip install captum")
        return None, None

    enc = trainer.tokenizer(
        text, max_length=trainer.cfg.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(trainer.device)
    attn_mask = enc["attention_mask"].to(trainer.device)

    style_raw = StylometricExtractor.extract(text)
    if trainer._style_scaler is not None and trainer._style_scaler.mean_ is not None:
        style_raw = trainer._style_scaler.transform([style_raw])[0]
    style = torch.tensor([style_raw], dtype=torch.float32).to(trainer.device)

    # Forward function for captum
    def forward_fn(input_ids_):
        out = trainer.model(input_ids_, attn_mask, style)
        # Return fake logit (class 1 = fake)
        return out["fake_logits"][:, 1]

    embeddings = trainer.model.encoder.embeddings
    lig = LayerIntegratedGradients(forward_fn, embeddings)

    attributions = lig.attribute(input_ids, n_steps=50)
    # Sum across embedding dimensions
    attr_sum = attributions.sum(dim=-1).squeeze(0)  # [seq_len]
    attr_sum = attr_sum / torch.norm(attr_sum)

    tokens = trainer.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    actual_len = attn_mask[0].sum().item()

    return tokens[:actual_len], attr_sum[:actual_len].cpu().numpy()


def plot_token_importance(tokens, scores, title, save_path, method="attention"):
    """Plot token importance as a heatmap bar chart."""
    # Remove special tokens and merge subwords
    clean_tokens = []
    clean_scores = []
    for t, s in zip(tokens, scores):
        if t in ("<s>", "</s>", "<pad>"):
            continue
        if t.startswith("▁"):
            clean_tokens.append(t[1:])
        else:
            # Subword: merge with previous
            if clean_tokens:
                clean_tokens[-1] += t
                clean_scores[-1] = max(clean_scores[-1], s)
                continue
            else:
                clean_tokens.append(t)
        clean_scores.append(s)

    # Limit to first 40 tokens for readability
    if len(clean_tokens) > 40:
        clean_tokens = clean_tokens[:40]
        clean_scores = clean_scores[:40]

    scores_arr = np.array(clean_scores)
    # Normalize to [0, 1]
    if scores_arr.max() > scores_arr.min():
        scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
    else:
        scores_norm = np.zeros_like(scores_arr)

    fig, ax = plt.subplots(figsize=(max(12, len(clean_tokens) * 0.4), 3))

    cmap = plt.cm.YlOrRd
    colors = [cmap(s) for s in scores_norm]

    bars = ax.bar(range(len(clean_tokens)), scores_norm, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(clean_tokens)))
    ax.set_xticklabels(clean_tokens, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"{method.capitalize()} Score")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def plot_text_heatmap(tokens, scores, title, save_path):
    """Plot text with color-coded background based on importance."""
    # Clean tokens
    words = []
    word_scores = []
    for t, s in zip(tokens, scores):
        if t in ("<s>", "</s>", "<pad>"):
            continue
        if t.startswith("▁"):
            words.append(t[1:])
            word_scores.append(s)
        else:
            if words:
                words[-1] += t
                word_scores[-1] = max(word_scores[-1], s)
            else:
                words.append(t)
                word_scores.append(s)

    scores_arr = np.array(word_scores)
    if scores_arr.max() > scores_arr.min():
        scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
    else:
        scores_norm = np.zeros_like(scores_arr)

    fig, ax = plt.subplots(figsize=(14, 2))
    ax.axis("off")

    x_pos = 0.02
    y_pos = 0.5
    cmap = plt.cm.YlOrRd

    for word, score in zip(words, scores_norm):
        color = cmap(score)
        bg_color = (*color[:3], 0.4)  # semi-transparent
        text_obj = ax.text(x_pos, y_pos, word + " ", fontsize=10,
                          transform=ax.transAxes, verticalalignment="center",
                          bbox=dict(boxstyle="round,pad=0.1", facecolor=bg_color, edgecolor="none"))

        # Get text width for positioning
        fig.canvas.draw()
        bb = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_ax = bb.transformed(ax.transAxes.inverted())
        x_pos += bb_ax.width + 0.005

        if x_pos > 0.95:
            x_pos = 0.02
            y_pos -= 0.4

    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


def main():
    print("=" * 62)
    print("  ATTENTION & INTERPRETABILITY ANALYSIS")
    print("=" * 62)
    print()
    print("  DISCLAIMER: Attention weights provide an indication of")
    print("  focus but do not necessarily reflect causal importance.")
    print("  (Jain & Wallace, 2019)")
    print()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig()
    trainer = Model1ExpertTrainer(cfg)
    trainer.build_model()
    trainer.load_best_model()

    for i, sample in enumerate(SAMPLE_TEXTS):
        print(f"\n{'─' * 62}")
        print(f"  Sample {i+1}: {sample['label']}")
        print(f"  Text: {sample['text'][:80]}...")

        # Get prediction
        pred = trainer.predict(sample["text"])
        print(f"  Prediction: {pred['fake_class']} ({pred['fake_score']:.1%}), "
              f"Manip={pred['manipulation_score']:.1%}, "
              f"Sent={pred['sentiment_class']}")

        # Attention weights
        tokens, attn_scores = get_attention_weights(trainer, sample["text"])
        plot_token_importance(
            tokens, attn_scores,
            f"Sample {i+1} ({sample['expected'].upper()}) — Attention Weights",
            PLOT_DIR / f"attention_sample_{i+1}.png",
            method="attention"
        )
        plot_text_heatmap(
            tokens, attn_scores,
            f"Sample {i+1}: {sample['label']}",
            PLOT_DIR / f"attention_heatmap_{i+1}.png"
        )

        # NOTE: Integrated Gradients (Captum) disabled — causes CUDA device-side
        # assert that corrupts GPU state for subsequent samples.
        # Attention weights are sufficient for thesis interpretability analysis.

    print(f"\n{'=' * 62}")
    print("  All plots saved to outputs/model1/plots/")
    print("  Use attention_heatmap_*.png in thesis for visual explanation")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
