# Model 1 — Multi-Task Text-Based Fake News Detection

**Current version:** v2.4.0 · **Last updated:** March 2026
**Backbone:** XLM-RoBERTa Base · **Tasks:** 4 (3 active, 1 passive) · **GNN interface:** 128-d manipulation embedding

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File Structure](#2-file-structure)
3. [Architecture — Full Data Flow](#3-architecture--full-data-flow)
4. [Configuration Reference](#4-configuration-reference-configsconfigpy)
5. [Model Architecture](#5-model-architecture-srcmodelstexttransformerpy)
6. [Loss Function](#6-loss-function-srctraininglosspy)
7. [Data Loading and Preprocessing](#7-data-loading-and-preprocessing)
8. [Anti-Shortcut Measures](#8-anti-shortcut-measures)
9. [Stylometric Features](#9-stylometric-features-srcfeaturesstylometrypy)
10. [Training Pipeline](#10-training-pipeline-srctrainingtext_trainerpy)
11. [GNN Feature Exporter](#11-gnn-feature-exporter-srcfeaturesgnn_exporterpy)
12. [Main Entry Point](#12-main-entry-point-mainpy)
13. [Checkpoint Structure](#13-checkpoint-structure)
14. [Data Contract — Sentinel Label System](#14-data-contract--sentinel-label-system)
15. [Design Decisions and Conscious Trade-offs](#15-design-decisions-and-conscious-trade-offs)
16. [Change History](#16-change-history)

---

## 1. Project Overview

Model 1 is a multi-task text classifier that analyzes a news text across four different dimensions. Four tasks share a single XLM-RoBERTa encoder and learn simultaneously:

| Task | Type | Output | Status |
|---|---|---|---|
| Fake News Detection | Binary classification | `fake_prob` in [0,1], `fake_class` in {real, fake} | Active |
| Sentiment Classification | 3-class classification | `sentiment_class` in {negative, neutral, positive} | Active |
| Manipulation Detection | Binary classification | `manipulation_score` in [0,1] | Active |
| Sentiment Intensity | Intensity regression (BCEWithLogitsLoss) | `sentiment_intensity` in [0,1] | Passive (loss zeroed) |

> **Sentiment Intensity Note:** The `sentiment_intensity` head's loss term has been zeroed since v2.0. Since target values are derived deterministically from class labels (neg->0.1, neu->0.5, pos->0.9), it carries no additional information beyond the class head. The head is preserved in the architecture (for inference) but excluded from the training loss.

The model also produces a **128-dimensional manipulation representation vector** (`manipulation_vector`) per article. This vector is used by Model 2 (GNN) as a graph node feature.

### Dataset Structure (v2.4)

| Task | Data Sources | Total Training Samples |
|---|---|---|
| Fake/Real | GossipCop + PolitiFact (2 domains) | ~33.3K |
| Manipulation | SemEval 2020 + MBIB + BABE | ~22.8K |
| Sentiment | NewsMTSC + tweet_eval + SST-5 | ~48.4K |

---

## 2. File Structure

```
FakeNews-Multimodal-System/
|
+-- main.py                          # CLI entry point (--train, --predict, --test)
|
+-- Makefile                         # Pipeline automation (fix-semeval, train, test)
|
+-- MODEL1_DOCUMENTATION.md          # This file
+-- CURRENT_STATUS.md                # Current status and training run summaries
+-- PROJECT_STATUS.md                # Full project status document
+-- MODEL1_FINALIZE.md               # Model 1 finalization report
|
+-- configs/
|   +-- __init__.py
|   +-- config.py                    # All constants and TrainerConfig dataclass
|
+-- scripts/
|   +-- threshold_tuning.py          # Post-hoc threshold & calibration tuning
|   +-- style_ablation.py            # Style feature ablation study
|   +-- adversarial_test.py          # Adversarial robustness test (15 pairs)
|   +-- attention_analysis.py        # Attention visualization (4 samples)
|   +-- error_analysis.py            # Misclassification pattern analysis
|   +-- head_ablation.py             # Full Model vs Only-Fake comparison
|
+-- src/
    +-- features/
    |   +-- __init__.py
    |   +-- stylometry.py            # Stylometric feature extraction + z-score normalizer
    |   +-- gnn_exporter.py          # Generates .pt files for GNN from trained model
    |
    +-- models/
    |   +-- __init__.py
    |   +-- text_transformer.py      # Main model: OptimizedMultiTaskModel
    |   +-- graph_net.py             # Model 2 (GNN) placeholder
    |   +-- fusion.py                # Fusion layer placeholder
    |
    +-- preprocessing/
    |   +-- __init__.py
    |   +-- data_loader.py           # SemEvalParser, PropagandaDataset, SimpleNewsDataset
    |   +-- graph_builder.py         # Graph structure builder
    |   +-- text_cleaner.py          # Source debiasing, style normalization, text cleaning
    |
    +-- training/
    |   +-- __init__.py
    |   +-- loss.py                  # MultiTaskLoss, AsymmetricBinaryLoss, BinaryFocalLoss
    |   +-- text_trainer.py          # Model1ExpertTrainer, run_full_pipeline
    |
    +-- utils/
        +-- __init__.py
        +-- common.py                # get_device(), set_seed()
        +-- metrics.py               # Evaluation helpers (stub)
```

### Expected Data Directory

```
data/
+-- train-articles/                  # SemEval articles (.txt)
+-- train-labels-task2-technique-classification/  # SemEval propaganda span labels
+-- dev-articles/                    # SemEval dev articles (75 articles)
+-- upfd/
    +-- news_content.json            # UPFD news content (for GNN feature export)

outputs/
+-- model1/
|   +-- best_model.pt                # Model weights + config (~1 GB)
|   +-- tokenizer/                   # XLM-R tokenizer files
|   +-- style_scaler.npz             # Fitted StyleScaler from training data
|   +-- training_history.json        # Epoch-level metrics
|   +-- test_results.json            # Test results
|   +-- plots/                       # Training plots + analysis plots
+-- gnn_features/
    +-- {news_id}.pt                 # Per-article feature files
    +-- index.json                   # news_id -> .pt file path mapping
    +-- feature_matrix.pt            # Bulk matrix of all vectors [N x 128]
```

---

## 3. Architecture — Full Data Flow

```
                         Raw Text
                             |
                   +-- debias_source() --------+  Anti-shortcut
                   +-- normalize_style() ------+  preprocessing
                             |
                    +--------v---------+
                    |  XLM-RoBERTa     |  (12 layers, 768-d)
                    |  Encoder         |
                    +--------+---------+
                             |
                    CLS vector (768-d) + dropout(0.1)
                    +--------+-----------------------------------+
                    |                    |                        |
           +--------v------+    +--------v--------+    +--------v------------+
           |  Fake Head    |    | Sentiment Head   |    | Manipulation Head   |
           |               |    |                  |    |  768->512->256->128  |
           |  style_proj   |    |  768->512->256   |    |  (3-layer bottleneck)|
           |  5->64        |    |  ->GELU->Drop    |    +--------+------------+
           |    +          |    |  ->256->3         |             |
           |  CLS(768)     |    |                  |    manipulation_embedding
           |  LayerNorm    |    |  Intensity: 768  |    (128-d) -> GNN Model 2
           |  832-d        |    |  ->64->1          |             |
           |  ->256->GELU  |    | (loss zeroed)    |    +--------v-----------+
           |  ->Drop(0.3)  |    +--------+---------+    | manipulation_      |
           |  ->256->2     |             |              | classifier (128->1)|
           +--------+------+    sentiment_logits        +--------+-----------+
                    |           sentiment_intensity               |
               fake_logits                              manipulation_logits
```

### Dimension Reference

| Tensor | Shape | Description |
|---|---|---|
| `input_ids` | `[B, 512]` | Token IDs (B = batch size) |
| `attention_mask` | `[B, 512]` | Padding mask |
| `style_feats` | `[B, 5]` | Raw stylometric features |
| `cls_vec` | `[B, 768]` | XLM-R CLS output (after dropout) |
| `style_proj` | `[B, 64]` | Projected style vector |
| `enhanced` | `[B, 832]` | `[cls_vec || style_proj]` + LayerNorm |
| `manip_hidden` | `[B, 128]` | Manipulation head intermediate representation |
| `fake_logits` | `[B, 2]` | Raw class logits |
| `manipulation_logits` | `[B]` | Raw binary logit |
| `manipulation_vector` | `[B, 128]` | Task-specific embedding for GNN |

---

## 4. Configuration Reference (`configs/config.py`)

### Module-Level Constants

| Constant | Value | Description |
|---|---|---|
| `MODEL_NAME` | `"xlm-roberta-base"` | HuggingFace model identifier |
| `HIDDEN_SIZE` | `768` | XLM-R hidden dimension |
| `STYLE_FEAT_DIM` | `5` | Number of raw stylometric features |
| `STYLE_PROJ_DIM` | `64` | Post-projection dimension (for fake_head input) |
| `ENHANCED_DIM` | `832` | `HIDDEN_SIZE + STYLE_PROJ_DIM` — fake_head input size |
| `MANIP_EMBED_DIM` | `128` | Manipulation embedding dimension (GNN node feature) |
| `MAX_SEQ_LEN` | `512` | Tokenizer maximum token length |
| `LABEL_SMOOTH_EPS` | `0.05` | CrossEntropyLoss label smoothing value |

### `TrainerConfig` Dataclass — Full Field List

#### Path Fields

| Field | Default | Description |
|---|---|---|
| `articles_dir` | `"data/train-articles"` | SemEval article texts |
| `labels_dir` | `"data/train-labels-task2-technique-classification"` | SemEval propaganda labels |
| `upfd_dir` | `"data/upfd"` | UPFD news JSON |
| `output_dir` | `"outputs/model1"` | Checkpoint, tokenizer, history save location |
| `gnn_output_dir` | `"outputs/gnn_features"` | GNN .pt files |

#### Model Fields

| Field | Default | Description |
|---|---|---|
| `model_name` | `MODEL_NAME` | Backbone to fine-tune |
| `max_seq_len` | `512` | Tokenizer truncation limit |

#### Training Hyperparameters

| Field | Default | Description |
|---|---|---|
| `num_epochs` | `12` | Maximum epoch count |
| `batch_size` | `32` | Visible batch size (fits GPU memory with MAX_SEQ_LEN=512) |
| `learning_rate` | `2e-5` | Base learning rate for task heads |
| `weight_decay` | `1e-2` | AdamW L2 regularization |
| `warmup_ratio` | `0.1` | 10% of total steps as linear warmup |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `fp16` | `True` | Mixed precision (AMP) |
| `gradient_accum` | `4` | Effective batch = `batch_size x gradient_accum = 128` |
| `early_stopping_patience` | `7` | Stop if composite score doesn't improve for N epochs |

#### Loss Weights

| Field | Default | Description |
|---|---|---|
| `lambda_fake` | `1.5` | Explicit fake head weight (prevents gradient starvation) |
| `lambda_sentiment` | `1.3` | Sentiment loss weight (boosted for 75% target) |
| `lambda_manipulation` | `1.5` | Manipulation loss weight (balanced with fake) |

#### Dataset Routing

| Field | Default | Description |
|---|---|---|
| `use_gossipcop` | `True` | GossipCop (primary fake/real, UPFD-compatible) |
| `gossipcop_max_per_class` | `12_000` | Cap GossipCop to balance with PolitiFact |
| `use_politifact` | `True` | PolitiFact (political domain fake/real) |
| `politifact_max_per_class` | `25_000` | Use all available PolitiFact data |
| `use_liar_fallback` | `False` | LIAR DISABLED for training (format conflict) |
| `use_liar_test` | `True` | Load LIAR for cross-domain test only |
| `use_welfake` | `False` | WELFake DISABLED (99.5% source bias artifact) |
| `use_isot_test` | `False` | ISOT DISABLED (style artifact) |
| `use_newsmtsc` | `True` | NewsMTSC (news domain sentiment) |
| `use_tweet_sentiment` | `True` | tweet_eval (secondary sentiment) |
| `tweet_sentiment_secondary_cap` | `12_000` | tweet_eval per-class cap |
| `use_sst5` | `True` | SST-5 (movie reviews, 5->3 class mapping) |
| `sst5_max_per_class` | `5_000` | SST-5 per-class cap |
| `use_manip_augmentation` | `True` | MBIB + BABE manipulation augmentation |
| `manip_aug_cap_per_class` | `5_000` | MBIB per-class cap |
| `use_proppy` | `True` | Proppy (HF — may not be loadable) |
| `use_financial_phrasebank` | `False` | financial_phrasebank DISABLED |

#### Other

| Field | Default | Description |
|---|---|---|
| `val_split` | `0.15` | Validation ratio (article-id based, no data leakage) |
| `seed` | `42` | Global seed for all randomness |
| `normalize_style` | `True` | StyleScaler z-score normalization enabled |
| `use_style_in_fake` | `True` | Fake head uses stylometric features (ablation flag) |
| `layer_lr_decay` | `0.95` | Layer-wise LR decay: `0.95^12 ~ 0.54x` for embeddings |
| `use_torch_compile` | `False` | `torch.compile()` (optional) |
| `use_focal_loss` | `True` | AsymmetricBinaryLoss for manipulation head |

---

## 5. Model Architecture (`src/models/text_transformer.py`)

### `OptimizedMultiTaskModel`

Multi-task transformer architecture built on XLM-RoBERTa Base. All tasks share the encoder; each task has its own specialized head.

#### `__init__` — Components

```python
self.encoder       # XLM-RoBERTa Base: ~125M parameters
self.dropout       # Dropout(0.1) — applied to CLS vector

# Style projection layer
self.style_proj    # Linear(5->64) + GELU
                   # Why needed: Raw 5-d features concatenated with 768-d CLS would
                   # only represent 0.6% of the gradient surface (5/773).
                   # 64-d projection raises this to 7.7% (64/832).

self.layer_norm    # LayerNorm(832) — on enhanced embedding

# Fake detection head (takes enhanced embedding: CLS + style)
self.fake_head     # Linear(832->256) + GELU + Dropout(0.3) + Linear(256->2)
                   # NOTE: Dropout 0.3 (other heads use 0.1) — overfitting prevention

# Sentiment heads (take CLS only) — 2 hidden layers for richer features
self.sentiment_class_head      # Linear(768->512) + GELU + Drop(0.1)
                               # + Linear(512->256) + GELU + Drop(0.1)
                               # + Linear(256->3)
self.sentiment_intensity_head  # Linear(768->64) + GELU + Dropout(0.1) + Linear(64->1)
                               # NOTE: Loss zeroed, kept for inference only

# Manipulation feature extractor + classifier (kept separate)
self.manipulation_feature      # Linear(768->512) + GELU + Drop(0.1)
                               # + Linear(512->256) + GELU + Drop(0.1)
                               # + Linear(256->128) + GELU + Drop(0.1)
                               # Three layers: single 768->128 projection (6x compression)
                               # would lose information; intermediate layers improve quality
self.manipulation_classifier   # Linear(128->1) — for loss only
```

#### `forward(input_ids, attention_mask, style_feats)` — Return Values

```python
{
    "fake_logits":           Tensor[B, 2],   # raw logits, before Softmax
    "sentiment_logits":      Tensor[B, 3],   # raw logits, before Softmax
    "sentiment_intensity":   Tensor[B],      # raw logit, before Sigmoid (loss zeroed)
    "manipulation_logits":   Tensor[B],      # raw logit, before Sigmoid
    "embeddings":            Tensor[B, 832], # enhanced (backward compatibility)
    "manipulation_embedding":Tensor[B, 128], # task-specific, sent to GNN
}
```

#### `get_predictions(input_ids, attention_mask, style_feats)` — Inference Mode

Wrapped with `@torch.no_grad()`. Converts logits to probabilities:

```python
{
    "fake_prob":           Tensor[B],    # Softmax(fake_logits)[:, 1]
    "fake_class":          Tensor[B],    # argmax -> 0=real, 1=fake
    "sentiment_prob":      Tensor[B, 3], # Softmax(sentiment_logits)
    "sentiment_class":     Tensor[B],    # argmax -> 0=neg, 1=neu, 2=pos
    "sentiment_intensity": Tensor[B],    # Sigmoid(sentiment_intensity_logit) -> [0,1]
    "manipulation_score":  Tensor[B],    # Sigmoid(manipulation_logits) -> [0,1]
    "manipulation_vector": Tensor[B, 128], # manipulation_embedding
}
```

---

## 6. Loss Function (`src/training/loss.py`)

### `MultiTaskLoss`

Multi-task loss function designed to work with heterogeneous datasets. Different datasets provide labels for different tasks; sentinel value `-1` causes the loss for that task to be zeroed for that batch.

#### Formula

```
total_loss = 1.5 * loss_fake
           + 1.3 * (loss_sentiment_class + 0)
           + 1.5 * loss_manipulation
```

> **Note:** `loss_sentiment_intensity` is always 0 since v2.0. Gradient budget is redirected to manipulation and fake heads.

Effective weights with current lambdas:

| Loss Term | Function | Lambda | Typical Scale |
|---|---|---|---|
| `loss_fake` | CrossEntropyLoss(label_smooth=0.05) | x1.5 | 0.3-0.7 |
| `loss_sentiment_class` | CrossEntropyLoss(label_smooth=0.05, weight=[1.0, 2.5, 2.5]) | x1.3 | 0.3-0.7 |
| `loss_sentiment_intensity` | BCEWithLogitsLoss | x1.3 | 0.0 (zeroed) |
| `loss_manipulation` | AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05) | x1.5 | 0.1-0.4 |

#### Sentiment Class Weights

```python
sent_weights = [1.0, 2.5, 2.5]  # [negative, neutral, positive]
# Symmetric boost for both neutral AND positive classes:
# - Neutral recall was ~55% (under-predicted)
# - Positive recall was ~65% (also under-predicted vs negative ~80%+)
# - Symmetric boost prevents the model from defaulting to negative
# register_buffer ensures automatic device tracking
```

#### Sentinel Masking

```python
fake_mask  = targets["fake_label"]        != -1   # Always False for SemEval
sent_mask  = targets["sentiment_label"]   != -1   # Always False for fake datasets
manip_mask = targets["manipulation_label"] != -1   # Always False for fake/sentiment datasets
# Samples where mask is False are completely excluded from that loss term
```

**Why all three masks are mandatory:** `BCEWithLogitsLoss` and `AsymmetricBinaryLoss` expect targets in `[0, 1]`. Passing `-1.0` float produces undefined gradients — causing silent incorrect training.

**Dataset -> mask mapping:**
- **SemEval + MBIB + BABE:** `manipulation_label` real, `fake_mask` and `sent_mask` False -> only manipulation head updates
- **GossipCop / PolitiFact:** `fake_label` real, `manip_mask` and `sent_mask` False -> only fake head updates
- **NewsMTSC / tweet_eval / SST-5:** `sentiment_label` real, `fake_mask` and `manip_mask` False -> only sentiment head updates
- Mixed batch: each sample only affects the head for which it has a label

#### `AsymmetricBinaryLoss` (Ridnik et al., 2021)

Replaces BinaryFocalLoss in v2.4. Unlike symmetric Focal Loss, ASL uses separate gamma values for positive and negative samples. Critical for imbalanced propaganda detection (~21% positive):

| Parameter | Value | Description |
|---|---|---|
| `gamma_pos` | `1` | Preserves gradient flow for hard positives (minority) |
| `gamma_neg` | `3` | Aggressively down-weights easy negatives (majority ~79%) |
| `clip` | `0.05` | Probability shifting: shifts negative probabilities, reducing contribution of very easy negatives |

#### `BinaryFocalLoss` (kept for reference/ablation)

```
FL(p_t) = alpha_t x (1 - p_t)^gamma x BCE(p_t)
```

| Parameter | Value | Description |
|---|---|---|
| `alpha` | `0.75` | More weight to positive (propaganda) class |
| `gamma` | `2.0` | Focus on hard examples (low confidence) |

---

## 7. Data Loading and Preprocessing

### Data Sources (v2.4)

#### Fake/Real (2 Training Domains + 1 Cross-Domain Test)

| Dataset | Source | Size | Domain | Preprocessing |
|---|---|---|---|---|
| GossipCop | HF: GonzaloA/fake_news | ~17.3K train | Entertainment/celebrity | `clean_for_fake_detection()` |
| PolitiFact | HF: Cartinoe5930/Politifact | ~16K train | Political fact-checks | `clean_for_fake_detection()` |
| LIAR | UCSB TSV (PolitiFact) | ~12.8K total | Political statements | TEST ONLY |

**LIAR 6-class -> Binary Mapping:**
- Real (0): true, mostly-true, half-true
- Fake (1): barely-true, false, pants-fire

**Removed:**
- WELFake: DISABLED. 99.5% accuracy = source-style artifact.
- ISOT: DISABLED. 0.1% accuracy = complete domain incompatibility.

#### Manipulation (SemEval + Augmentation)

| Dataset | Source | Size | Description |
|---|---|---|---|
| SemEval 2020 Task 11 | Local article files | ~9,679 sentences (371 articles) | Primary benchmark |
| MBIB linguistic_bias | HF | 5,000 pos + 5,000 neg (cap) | Augmentation |
| BABE media_bias | HF | ~3,120 sentences | Augmentation |

Total: ~22,799 training sentences. MBIB + BABE are only added to train split — val/test remain pure SemEval.

#### Sentiment (News + Tweet + SST-5)

| Dataset | Source | Size | Description |
|---|---|---|---|
| NewsMTSC | GitHub JSONL (direct download) | ~8,739 train | News domain (primary) |
| tweet_eval | HF | ~31,093 train (12K/class cap) | Linguistic diversity (secondary) |
| SST-5 | HF: SetFit/sst5 | ~8,544 train (5K/class cap) | Movie reviews, 5->3 class mapping |

Total: ~48,376 training samples.

> **NewsMTSC Note:** HF `trust_remote_code` support ended, so JSONL files are downloaded directly from GitHub. Label mapping: negative->0, neutral->1, positive->2.

### Dataset Classes

#### `SentenceSample` — Data Unit (SemEval)

```python
@dataclass
class SentenceSample:
    article_id: str    # Article ID (e.g., "123456")
    sentence_id: int   # Sentence order within article
    text: str          # Sentence text
    label: int         # 0=normal, 1=contains propaganda/manipulation
    char_start: int    # Character start in original article
    char_end: int      # Character end in original article
```

#### `PropagandaDataset`

Converts SemEval sentence samples to model input.

```python
{
    "input_ids":           Tensor[512],   # token IDs
    "attention_mask":      Tensor[512],   # padding mask
    "style_feats":         Tensor[5],     # (normalized) stylometric features
    "manipulation_label":  Tensor,        # float32, 0.0 or 1.0
    "fake_label":          Tensor,        # long, -1 (not available in SemEval)
    "sentiment_label":     Tensor,        # long, -1 (not available in SemEval)
    "sentiment_intensity": Tensor,        # float32, 0.0 (placeholder)
}
```

#### `SimpleNewsDataset`

For GossipCop, PolitiFact, LIAR, NewsMTSC, tweet_eval, SST-5 and similar. **Schema is identical to PropagandaDataset** — can be combined with `ConcatDataset` without custom `collate_fn`.

---

## 8. Anti-Shortcut Measures

Since v2.0, comprehensive measures ensure the model learns news content rather than source style.

### 8.1 Source Debiasing (`text_cleaner.py: debias_source()`)

60+ news source names are masked, bylines and datelines are removed:

| Category | Examples | Action |
|---|---|---|
| Source names | Reuters, CNN, TMZ, GossipCop, Breitbart, PolitiFact... | -> `[SOURCE]` |
| Byline | "By John Smith", "Reported by Jane Doe for CNN" | Removed |
| Dateline | "NEW YORK (Reuters) —" | Removed |
| URL | https://... | -> `[LINK]` |

### 8.2 Style Normalization (`text_cleaner.py: normalize_style()`)

| Rule | Example | Result |
|---|---|---|
| Repeated punctuation | `!!!` | `!` |
| ALL CAPS (4+ letters) | `BREAKING NEWS` | `Breaking News` |
| Extra whitespace | multiple spaces | single space |

### 8.3 Multi-Domain Training

Two different domains to prevent single-domain lock-in:
- **GossipCop:** Entertainment/celebrity gossip
- **PolitiFact:** Political fact-checks

### 8.4 Disabled Datasets

- **WELFake:** 99.5% accuracy = source-style artifact
- **ISOT:** 0.1% accuracy = complete domain incompatibility

### 8.5 Combined Pipeline

```python
def clean_for_fake_detection(text: str) -> str:
    text = debias_source(text)      # Source name, byline, dateline, URL
    text = normalize_style(text)    # ALL CAPS, repeated punctuation
    return text
```

---

## 9. Stylometric Features (`src/features/stylometry.py`)

### `StylometricExtractor`

Produces a 5-dimensional feature vector per text.

| Index | Feature | Formula | Why |
|---|---|---|---|
| 0 | `punct_density` | `(! + ?) / n_words` | Emotional manipulation signal |
| 1 | `caps_ratio` | `UPPERCASE / n_alpha_chars` | Headline/sensationalism indicator |
| 2 | `quotation_flag` | `1.0` if text contains quotes | Quote/source claim |
| 3 | `ttr` | `unique_words / n_words` | Word diversity; fake news tends to use repetitive language |
| 4 | `avg_sent_len_norm` | `(n_words / n_sentences) / 20.0` | Discourse-level complexity |

> **Style Ablation Result (v2.4):** Zeroing all style features causes NO performance drop. The encoder captures stylistic information implicitly through contextual embeddings.

### `StyleScaler`

Z-score normalizer fitted on training corpus. Saved to checkpoint to prevent train/inference inconsistency.

---

## 10. Training Pipeline (`src/training/text_trainer.py`)

### `Model1ExpertTrainer`

Main class managing all training, evaluation, and inference operations.

#### Data Loading (`load_data`)

```
1. SemEval articles read + MBIB + BABE augmentation added
2. GossipCop loaded from HF + clean_for_fake_detection()
3. PolitiFact loaded from HF + clean_for_fake_detection()
4. LIAR loaded from TSV (test only)
5. NewsMTSC loaded from GitHub JSONL
6. tweet_eval loaded from HF (12K/class cap)
7. SST-5 loaded from HF (5K/class cap, 5->3 class mapping)
8. All train splits combined with ConcatDataset
9. StyleScaler fitted on all train samples
10. WeightedRandomSampler: manages class imbalance (normalized per-split)
11. DataLoader: train (shuffle=sampler), val (shuffle=False)
```

#### Optimizer: Layer-wise LR Decay (`_build_optimizer`)

When `layer_lr_decay < 1.0` (default 0.95), each encoder layer gets a different learning rate:

| Component | LR Multiplier | Effective LR (base=2e-5) |
|---|---|---|
| Embeddings | `0.95^12 ~ 0.54` | ~1.08e-5 |
| Layer 0 (bottom) | `0.95^11 ~ 0.57` | ~1.13e-5 |
| Layer 5 (middle) | `0.95^6 ~ 0.74` | ~1.48e-5 |
| Layer 11 (top) | `0.95^0 = 1.00` | 2.00e-5 |
| Task heads | 1.00 | 2.00e-5 |

**Motivation:** Lower layers hold general linguistic knowledge and should not be over-updated (catastrophic forgetting risk). Upper layers and heads get full LR for task-specific adaptation.

#### Training Loop (`train`)

```
For each epoch:
    Model set to train mode
    For each batch:
        Forward pass (AMP with fp16)
        MultiTaskLoss computed (with sentinel masking)
        loss / gradient_accum for scaling
        Gradient accumulation (gradient_accum=4)
        Every 4 steps: unscale -> clip -> step -> scheduler

    Validation evaluation (_evaluate)

    Composite score computed:
        composite = (fake_f1 + manipulation_f1 + sentiment_acc) / 3

    If improved: checkpoint saved
    If not improved: epochs_no_improve incremented

    If early_stopping_patience exceeded: training stopped

training_history.json saved
```

#### Test Pipeline (`test`)

After training, all test splits are evaluated:

| Test | Description |
|---|---|
| GossipCop test | In-domain fake/real accuracy |
| PolitiFact test | Cross-domain fake/real accuracy |
| LIAR test | Cross-domain generalization (never trained) |
| SemEval test | Manipulation F1 (with threshold sweep) |
| Sentiment test | tweet_eval + NewsMTSC + SST-5 accuracy |

**Threshold Sweep:** Manipulation head threshold searched in [0.15, 0.75] range with 0.01 steps. Only on validation set (no test leak).

---

## 11. GNN Feature Exporter (`src/features/gnn_exporter.py`)

### `GNNFeatureExporter`

Runs trained Model 1 on UPFD news corpus and generates feature files for Model 2 (GNN).

#### Steps

```
1. data/upfd/news_content.json is read
   -> {"news_id": {"title": ..., "text": ...}} format

2. predict_batch() runs inference in 256-sample batches

3. For each article, outputs/gnn_features/{news_id}.pt is saved:
   {
       "news_id": str,
       "manipulation_vector": Tensor[128],  # task-specific embedding
       "fake_score": float,
       "sentiment_class": int,
       "sentiment_intensity": float,
       "manipulation_score": float,
   }

4. index.json: news_id -> .pt file path mapping

5. feature_matrix.pt: bulk [N x 128] matrix of all vectors
   {"news_ids": [...], "vectors": Tensor[N, 128], "dim": 128}
```

#### `manipulation_vector` Semantics

The 128-d vector sent to GNN comes from the `manipulation_feature` layer. Properties:
- Only receives gradients from `manipulation_classifier`'s loss (manipulation-specific)
- But input material is the 4-task CLS (multi-task representation richness preserved)
- Correct terminology: "manipulation-supervised projection of a multi-task representation"

---

## 12. Main Entry Point (`main.py`)

```bash
# Full pipeline: load data -> train -> load best model -> export GNN features
python main.py --train

# Test pipeline: load best model -> evaluate all test splits
python main.py --test

# Single text inference (default sample text)
python main.py --predict

# Custom text inference
python main.py --predict --text "The government is hiding that vaccines are dangerous!"
```

---

## 13. Checkpoint Structure

```
outputs/model1/
+-- best_model.pt
|   +-- epoch: int             # Best epoch number
|   +-- f1: float              # Best composite score
|   +-- model_state: dict      # state_dict() — all weights
|   +-- config: TrainerConfig  # Training configuration
|
+-- style_scaler.npz
|   +-- mean: ndarray[5]       # Mean per feature
|   +-- std: ndarray[5]        # Std per feature
|
+-- tokenizer/
|   +-- sentencepiece.bpe.model
|   +-- tokenizer.json
|   +-- ...
|
+-- training_history.json
+-- test_results.json
+-- plots/
```

> **Critical:** Without `style_scaler.npz`, a loaded model will see unnormalized features during inference — causing silent errors. StyleScaler must always be saved with the checkpoint.

---

## 14. Data Contract — Sentinel Label System

Different datasets have different label coverage. Sentinel value `-1` expresses which task is valid for which dataset:

| Dataset | `fake_label` | `manipulation_label` | `sentiment_label` |
|---|---|---|---|
| SemEval 2020 + MBIB + BABE | `-1` (N/A) | `0` or `1` | `-1` (N/A) |
| GossipCop / PolitiFact | `0` or `1` | `-1` (N/A) | `-1` (N/A) |
| NewsMTSC / tweet_eval / SST-5 | `-1` (N/A) | `-1` (N/A) | `0`, `1` or `2` |

`MultiTaskLoss` automatically excludes samples with `-1` labels from the corresponding loss term. This mechanism allows datasets from different sources to be combined with `torch.utils.data.ConcatDataset` without a custom `collate_fn`.

---

## 15. Design Decisions and Conscious Trade-offs

### CLS Token with 3 Active Tasks

All tasks share the same CLS vector. Disadvantage: each task's gradient pulls the encoder in different directions (gradient conflict). Advantage: tasks are thematically related and the shared representation benefits from multiple signals. Current regularization stack (LLRD + Dropout + weight_decay + label_smoothing + gradient_clip) is sufficient to manage this risk.

### WeightedRandomSampler + AsymmetricLoss Combination

Sampler provides data-level balancing, ASL provides loss-level balancing. gamma_neg=3 aggressively down-weights easy negatives; gamma_pos=1 preserves gradient flow for hard positives. clip=0.05 provides additional probability shifting for very easy negatives.

### Multi-Domain Training Strategy

Single-domain training leads to source-style shortcuts (GossipCop 97.6% + ISOT 0.46% as proof). Two domains (entertainment + political) force the model to focus on content truthfulness rather than source style.

### Sentiment Intensity Loss Zeroing

Sentiment intensity head targets are derived deterministically from class labels (neg->0.1, neu->0.5, pos->0.9), carrying no additional information. Zeroing the loss redirects gradient budget to manipulation and fake heads.

### `manipulation_embedding` — Why Not Pure Task-Specific?

The 128-d vector's input material is the 4-task CLS. Therefore it should be called "manipulation-supervised projection" rather than "manipulation-specific." For GNN this may be an advantage: the vector carries both manipulation information and general text understanding.

### Fake Head Dropout (0.3 vs 0.1)

Fake head uses higher dropout (0.3) than other heads. This provides heavier regularization since fake detection is more susceptible to source-style shortcuts.

---

## 16. Change History

### v2.4.0 — March 2026 (Secenek A: Architectural + Data Improvements)

**`configs/config.py`:**
- `num_epochs` 10 -> 12
- `lambda_sentiment` 0.7 -> 1.3 (boosted for 75% target)
- `lambda_manipulation` kept at 1.5 (balanced with ASL)
- `tweet_sentiment_secondary_cap` 5_000 -> 12_000
- `use_sst5 = True` added (SST-5 dataset)
- `sst5_max_per_class = 5_000` added

**`src/training/loss.py`:**
- `AsymmetricBinaryLoss` class added (Ridnik et al., 2021)
- Manipulation loss switched from `BinaryFocalLoss(alpha=0.70, gamma=1.5)` to `AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05)`
- `sent_weights` changed from `[1.0, 3.0, 1.2]` to `[1.0, 2.5, 2.5]` (symmetric boost)

**`src/models/text_transformer.py`:**
- Sentiment head expanded: `768->256->3` to `768->512->256->3` (2 hidden layers)
- Manipulation head expanded: `768->256->128` to `768->512->256->128` (wider)

**`src/training/text_trainer.py`:**
- SST-5 dataset loader added (`SetFit/sst5`, 5-class -> 3-class mapping)
- Threshold sweep widened: `[0.30, 0.60]` to `[0.15, 0.75]`
- WeightedRandomSampler per-split normalization fixed

**Results:** FakeAcc=98.10%, ManipF1=0.6352 (+0.02), SentAcc=68.25% (-0.78%)

---

### v2.3.0 — March 2026 (PolitiFact + Lambda Tuning)

- PolitiFact added as second training domain
- LIAR and ISOT removed from training
- `lambda_fake = 1.5` added (explicit weight)
- GossipCop capped at 12K/class
- Post-hoc analysis complete (6 scripts)

**Results:** FakeAcc=98.15%, ManipF1=0.6139, SentAcc=69.03%

---

### v2.2.0 — March 2026 (Multi-Domain Training + ISOT)

- ISOT added to training (5K/class) + cross-domain test
- Threshold sweep range widened

---

### v2.1.0 — March 2026 (Source Debiasing + Augmentation)

- `debias_source()` added: 60+ source names masked
- MBIB + BABE augmentation added (~13K extra manipulation sentences)
- `lambda_manipulation` raised to 2.5
- BinaryFocalLoss tuned (alpha=0.65, gamma=2.0)
- `sent_weights` adjusted for neutral boost

---

### v2.0.0 — March 2026 (GossipCop Migration + NewsMTSC)

- WELFake DISABLED (99.5% accuracy = source-style artifact)
- GossipCop became primary fake/real dataset (UPFD-compatible)
- LIAR added for political domain generalization
- NewsMTSC added for news-domain sentiment
- `MAX_SEQ_LEN` 256 -> 512
- Sentiment intensity loss zeroed
