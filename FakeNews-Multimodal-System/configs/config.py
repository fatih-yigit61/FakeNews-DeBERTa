from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "microsoft/deberta-v3-base"
HIDDEN_SIZE = 768  # DeBERTa-v3-base also uses 768 hidden dim
STYLE_FEAT_DIM = 5          # punct_density, caps_ratio, quotation_flag, ttr, avg_sent_len_norm
STYLE_PROJ_DIM = 64         # stylometric features projected before CLS concat to ensure gradient reach
ENHANCED_DIM = 832          # HIDDEN_SIZE + STYLE_PROJ_DIM (768 + 64)
MANIP_EMBED_DIM = 128       # manipulation head hidden layer dim — used as GNN node feature
MAX_SEQ_LEN = 256           # reduced from 512: DeBERTa disentangled attn is O(n²), 256 is 4x faster
LABEL_SMOOTH_EPS = 0.05   # reduced from 0.10: too aggressive for multi-domain training (3 conflicting domains)


@dataclass
class TrainerConfig:
    articles_dir: str = "data/train-articles"
    labels_dir: str = "data/train-labels-task2-technique-classification"
    upfd_dir: str = "data/upfd"
    output_dir: str = "outputs/model1_deberta"
    gnn_output_dir: str = "outputs/gnn_features_deberta"

    model_name: str = MODEL_NAME
    max_seq_len: int = MAX_SEQ_LEN

    num_epochs: int = 12
    batch_size: int = 32           # DeBERTa disentangled attention needs more VRAM than XLM-R
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = False              # DeBERTa-v3 FP16 params incompatible with GradScaler — use bf16 instead
    bf16: bool = True               # BF16 autocast: A100 native, same speed as FP16 but no NaN (FP32 exponent range)
    gradient_accum: int = 4        # batch=32 * accum=4 = effective 128

    lambda_fake: float = 1.5          # explicit fake head weight (was implicit 1.0)
    lambda_sentiment: float = 1.3     # raised from 1.0: sentiment needs more gradient (69% → 75% target)
    lambda_manipulation: float = 1.5  # balanced with fake; ASL replaces focal loss for better precision-recall
    val_split: float = 0.15
    seed: int = 42

    # ── Fake/Real dataset routing ────────────────────────────────────────────
    # GossipCop (primary): fact-checked by GossipCop.com, aligns with UPFD graph.
    # Less source-style bias than WELFake. HF candidates tried in order.
    use_gossipcop: bool = True
    gossipcop_max_per_class: int = 12_000  # cap GossipCop to boost PolitiFact ratio in training
    # LIAR: PolitiFact 12.8K statements, 6-class → binary mapping.
    # DISABLED for training: short political statements conflict with full-article
    # datasets (GossipCop/ISOT), causing prediction inversion (ROC-AUC < 0.50).
    # Kept as cross-domain TEST to evaluate generalization.
    use_liar_fallback: bool = False
    use_liar_test: bool = True  # load LIAR for cross-domain test even when not training
    # PolitiFact (FakeNewsNet): political domain counterpart of GossipCop.
    # Same framework (Shu et al.), full articles, fact-check-based labels.
    # Compatible format prevents the dataset conflict seen with LIAR.
    use_politifact: bool = True
    politifact_max_per_class: int = 25_000  # raised: use all available PolitiFact data for 85%+ target
    # WELFake: DISABLED by default — %99.5 accuracy is source-style artifact.
    # Keep code for ablation/comparison, but don't use in primary training.
    use_welfake: bool = False

    # WELFake config (kept for ablation when use_welfake=True)
    welfake_csv: str = "data/WELFake_Dataset.csv"
    welfake_max_per_class: int = 15_000
    use_title_in_fake: bool = False
    welfake_topic_clusters: int = 50

    # ── Sentiment data sources ────────────────────────────────────────────────
    # NewsMTSC: ~11K news sentences with sentence-level sentiment annotations.
    # Best domain match for news-focused sentiment analysis.
    # Labels mapped to: 0=negative, 1=neutral, 2=positive.
    use_newsmtsc: bool = True
    newsmtsc_hf_path: str = "fhamborg/news_sentiment_newsmtsc"

    # financial_phrasebank: DISABLED — finance domain doesn't generalize to general news.
    use_financial_phrasebank: bool = False

    # tweet_eval: secondary source for volume + linguistic diversity.
    # Cap reduced to 3,000/class (from 15,000) since financial_phrasebank is primary.
    use_tweet_sentiment: bool = True
    tweet_sentiment_secondary_cap: int = 12_000  # raised from 8K: more sentiment data for 75% target
    tweet_sentiment_max_per_class: int = 7_000  # per class when news-domain source fails (fallback)

    # SST-5 (Stanford Sentiment Treebank): 11.8K sentences, 5-class → 3-class.
    # High-quality human-annotated sentiment with fine-grained labels.
    # Adds domain diversity (movie reviews) to complement news + tweet.
    use_sst5: bool = True
    sst5_max_per_class: int = 5_000   # cap per class to prevent dominating sentiment data

    # Proppy / external propaganda dataset for manipulation_head augmentation.
    # Tried after SemEval; adds article-level propaganda labels as additional
    # binary manipulation training signal.  Set False to skip.
    use_proppy: bool = True

    # MBIB + BABE augmentation for manipulation_head.
    # MBIB linguistic_bias (401K) capped per class; BABE (3.1K) fully used.
    # Only added to train split — val/test remain pure SemEval.
    use_manip_augmentation: bool = True
    manip_aug_cap_per_class: int = 5_000  # cap per class from MBIB linguistic_bias

    gnn_batch_size: int = 256
    normalize_style: bool = True
    # Ablation flag: whether fake_head uses stylometric features (CLS+style=832)
    # or only CLS (768). Set False to test without stylistic shortcut.
    use_style_in_fake: bool = True
    # Apply style normalization to WELFake text (reduce ALL CAPS, repeated punctuation).
    normalize_welfake_style: bool = True
    layer_lr_decay: float = 0.95    # 0.95^12 ≈ 0.54× for embeddings; less aggressive than 0.9
    use_torch_compile: bool = False
    use_focal_loss: bool = True     # BinaryFocalLoss for manipulation head (hard-example mining)
    early_stopping_patience: int = 7  # stop if composite score doesn't improve for N epochs (raised from 5 for multi-domain)

    # ── Held-out test splits ──────────────────────────────────────────────────
    # WELFake: 10% carved out before train/val split (80/10/10).
    # SemEval: article IDs split 3-way (70/15/15); dev-articles/ is also available.
    # tweet_eval: uses HuggingFace's native 'test' split (no manual carve-out needed).
    welfake_test_split: float = 0.10   # fraction of WELFake held out as test (applied first)

    # ISOT Fake News Dataset — used ONLY as held-out cross-domain test set (never in training).
    # HuggingFace cleaned version (Reuters bylines stripped): Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022
    # label convention: 'target' column, 0=real / 1=fake (same as WELFake).
    # Set to False to skip (e.g., on environments without internet access after initial run).
    use_isot_test: bool = False
    isot_hf_path: str = "Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022"
    isot_train_cap: int = 0           # 0 = ISOT test-only (no training). Set >0 for multi-domain training.
    isot_test_max: int = 5_000       # max ISOT samples in test set (None = use all ~6,822 val rows)


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([TrainerConfig])
