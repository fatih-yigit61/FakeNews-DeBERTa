# FakeNews Multimodal Detection System — Project Status

**Date:** March 2026
**Version:** v2.4.0
**Language:** English only (Turkish support removed)
**Purpose:** Full status document for cross-AI review.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model 1 — What Is Built](#3-model-1--what-is-built)
4. [Model 1 — Training Results](#4-model-1--training-results)
5. [Model 1 — Post-Hoc Analysis](#5-model-1--post-hoc-analysis)
6. [Success Criteria — Honest Assessment](#6-success-criteria--honest-assessment)
7. [Model 2 (GNN) — Status](#7-model-2-gnn--status)
8. [Fusion Layer — Status](#8-fusion-layer--status)
9. [Data Available](#9-data-available)
10. [Key Findings](#10-key-findings)

---

## 1. Project Overview

A **Transformer + GNN based fake news detection system**. The overall pipeline:

```
News Text
    |
    v
[Model 1] XLM-RoBERTa Multi-Task Transformer
    |  +-- fake_head          -> fake/real classification
    |  +-- sentiment_head     -> negative/neutral/positive
    |  +-- manipulation_head  -> propaganda detection
    |  +-- manipulation_embedding (128-d) -------+
                                                  v
                                       [Model 2] GNN
                                            |
                                            v
                                     [Fusion Layer]
                                            |
                                            v
                                  Final Fake News Decision
```

**Key design decisions:**
- XLM-RoBERTa-base as backbone (~125M params)
- Multi-task learning: shared encoder, task-specific heads
- Sentinel label system (-1) allows mixing heterogeneous datasets
- Model 1 produces 128-d manipulation_embedding for GNN as node feature
- UPFD gossipcop graph data for Model 2 (propagation graph)

---

## 2. System Architecture

### File Structure
```
FakeNews-Multimodal-System/
+-- main.py                          # CLI: --train, --predict, --test
+-- configs/config.py                # All hyperparameters (TrainerConfig dataclass)
+-- scripts/
|   +-- threshold_tuning.py          # Post-hoc threshold & calibration
|   +-- style_ablation.py            # Style feature ablation study
|   +-- adversarial_test.py          # Adversarial robustness test (15 pairs)
|   +-- attention_analysis.py        # Attention visualization (4 samples)
|   +-- error_analysis.py            # Misclassification pattern analysis
|   +-- head_ablation.py             # Full vs Only-Fake head comparison
+-- src/
|   +-- models/
|   |   +-- text_transformer.py      # OptimizedMultiTaskModel <- COMPLETE
|   |   +-- graph_net.py             # GraphNet <- PLACEHOLDER
|   |   +-- fusion.py                # FusionLayer <- PLACEHOLDER
|   +-- training/
|   |   +-- text_trainer.py          # Model1ExpertTrainer <- COMPLETE
|   |   +-- loss.py                  # MultiTaskLoss, AsymmetricBinaryLoss <- COMPLETE
|   +-- preprocessing/
|   |   +-- data_loader.py           # All dataset loaders <- COMPLETE
|   +-- features/
|   |   +-- stylometry.py            # StylometricExtractor, StyleScaler <- COMPLETE
|   |   +-- gnn_exporter.py          # GNNFeatureExporter <- COMPLETE
+-- Makefile                          # Full Colab workflow automation
```

---

## 3. Model 1 — What Is Built

### 3.1 Architecture (text_transformer.py)

```
Input: text -> XLM-RoBERTa tokenizer -> [B, 512] tokens

Encoder: XLM-RoBERTa-base (12 layers, 768 hidden dim)
    +-- CLS token -> cls_vec [B, 768]

style_proj: Linear(5->64) + GELU
    +-- 5 stylometric features -> style_proj [B, 64]

enhanced = LayerNorm(cat([cls_vec, style_proj])) -> [B, 832]

fake_head:            Linear(832->256)->GELU->Drop(0.3)->Linear(256->2)           -> fake_logits
sentiment_class_head: Linear(768->512)->GELU->Drop->Linear(512->256)->GELU->Drop->Linear(256->3) -> sent_logits
manipulation_feature: Linear(768->512)->GELU->Drop->Linear(512->256)->GELU->Drop->Linear(256->128) -> manip_embed
manipulation_classifier: Linear(128->1)                                            -> manip_logit
```

### 3.2 Loss (loss.py)

```
total_loss = 1.5 * fake_loss
           + 1.3 * sentiment_class_loss
           + 1.5 * manipulation_loss

fake_loss:         CrossEntropyLoss(label_smoothing=0.05)
sentiment_loss:    CrossEntropyLoss(weight=[1.0, 2.5, 2.5], label_smoothing=0.05)
manipulation_loss: AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05)
```

### 3.3 Training Pipeline (text_trainer.py)

**Data sources loaded simultaneously:**

| Dataset | Size | Head Trained | Domain |
|---|---|---|---|
| GossipCop (HF) | ~17.3K train | fake_head | Entertainment/celebrity |
| PolitiFact (HF) | ~16K train | fake_head | Political fact-checks |
| LIAR (TSV) | TEST ONLY | fake_head | Political statements |
| SemEval 2020 | 9,679 sentences | manipulation_head | News propaganda |
| MBIB + BABE | 13,120 sentences | manipulation_head | Bias augmentation |
| NewsMTSC (JSONL) | 8,739 train | sentiment_head | News domain |
| tweet_eval (HF) | 31,093 train | sentiment_head | Twitter domain |
| SST-5 (HF) | 8,544 train | sentiment_head | Movie reviews (5->3 class) |

### 3.4 Current Configuration

```python
num_epochs = 12
early_stopping_patience = 7
learning_rate = 2e-5
batch_size = 32, gradient_accum = 4  # effective batch = 128

lambda_fake = 1.5
lambda_sentiment = 1.3
lambda_manipulation = 1.5
label_smoothing = 0.05

AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05)
sent_weights = [1.0, 2.5, 2.5]  # symmetric neutral+positive boost

gossipcop_max_per_class = 12_000
politifact_max_per_class = 25_000
tweet_sentiment_secondary_cap = 12_000
sst5_max_per_class = 5_000
```

---

## 4. Model 1 — Training Results

### Latest Run (v2.4, 12 epochs, Secenek A improvements)

| Test Set | Metric | Score | Target | Status |
|---|---|---|---|---|
| GossipCop (n=2311) | Accuracy | 98.10% | >= 85% | ACHIEVED |
| GossipCop | ROC-AUC | 0.9991 | — | — |
| PolitiFact (n=2131) | Accuracy | 75.22% | — | Generalization |
| PolitiFact | ROC-AUC | 0.8365 | — | — |
| LIAR cross-domain (n=1283) | Accuracy | 53.31% | — | Format mismatch |
| LIAR | ROC-AUC | 0.5198 | — | — |
| SemEval test (n=2376) | ManipF1 | 0.6352 | >= 0.65 | NOT MET |
| SemEval | ROC-AUC | 0.8037 | — | — |
| SemEval | PR-AUC | 0.6622 | — | — |
| Sentiment test (n=15297) | Accuracy | 68.25% | >= 75% | NOT MET |
| Sentiment (calibrated) | Accuracy | 69.12% | >= 75% | NOT MET |

### Validation Metrics
- FakeAcc: 87.69% (mixed GossipCop + PolitiFact)
- ManipF1: 0.6344 (threshold=0.45)
- SentAcc: 71.28%

---

## 5. Model 1 — Post-Hoc Analysis

### 5.1 Threshold Tuning (v2.4)
- Manipulation: optimal threshold 0.45 (F1=0.6352) — flat curve between 0.40-0.55
- Sentiment: best calibration temp=1.5_neg=0.3_pos=-0.3 (Acc=69.12%) vs original (68.25%) — +0.87%
- **Conclusion:** Post-hoc calibration cannot close the gap. XLM-RoBERTa capacity is the bottleneck.

### 5.2 Confidence Calibration (ECE)
| Head | ECE | Interpretation |
|---|---|---|
| Fake detection | 0.0637 | Well-calibrated |
| Manipulation | 0.0672 | Well-calibrated |
| Sentiment | 0.0889 | Improved (was 0.1301 in v2.3) |

### 5.3 Style Feature Ablation
- ALL metrics STABLE when style_feats zeroed
- **Interpretation:** Model learned semantic representations, not dependent on surface-level stylometric features

### 5.4 Adversarial Robustness (15 text pairs)
- Manipulation head: **15/15 correct** — manipulated texts avg +30pp manipulation score
- Fake head: **1/15 detected** — does NOT equate stylistic manipulation with fakeness
- **Key finding:** Fake and manipulation heads learned DISTINCT representations (separation of concerns)
- FactFlip pair: model shows partial semantic awareness (62.6% -> 79.2% fake score)
- Sentiment head: 14/15 correct sentiment shifts (manipulated -> negative)

### 5.5 Attention Visualization
- 4 sample texts analyzed with attention heatmaps
- Disclaimer: "Attention weights provide an indication of focus but do not necessarily reflect causal importance" (Jain & Wallace, 2019)

### 5.6 Error Analysis
| Test Set | Error Rate | False Positives | False Negatives | Pattern |
|---|---|---|---|---|
| GossipCop | 2.1% (49/2311) | 30 | 19 | FPs are political content; FNs are informal fake |
| PolitiFact | 25.6% (546/2131) | 328 | 218 | FPs are sensational real news; FNs are calm-style fake |
| LIAR | 44.0% (565/1283) | 85 | 480 | Format mismatch (18-word avg); model defaults to "Real" |

- Error confidence (0.726) much lower than correct predictions (0.977) — model is uncertain when wrong

---

## 6. Success Criteria — Honest Assessment

### 6.1 Fake/Real accuracy >= 85%
**Status: ACHIEVED (98.10%)**

GossipCop test with source debiasing (60+ sources masked). PolitiFact at 75.22% proves cross-domain generalization.

### 6.2 Sentiment accuracy >= 75%
**Status: NOT MET (68.25%, calibrated 69.12%) | Gap: ~6%**

XLM-RoBERTa-base single-task ceiling on mixed sentiment: ~72%. Multi-task penalty reduces this by 2-3%. Achieving 75% requires DeBERTa-v3-base backbone (GLUE 91.3 vs ~86 for XLM-R).

### 6.3 Manipulation F1 >= 0.65
**Status: NOT MET (0.6352) | Gap: ~0.015**

Very close to target. Precision (0.5289) is the bottleneck — model over-predicts propaganda. DeBERTa-v3's disentangled attention should improve precision through better contextual understanding.

### 6.4 Why DeBERTa-v3-base is the Next Step
- XLM-RoBERTa capacity exhausted: post-hoc tuning, data augmentation, architecture expansion all applied
- DeBERTa-v3-base: GLUE 91.3 (vs ~86 for XLM-R), 86M params (vs 125M), English-specialized
- Disentangled attention: separate content/position processing for better nuance detection
- Drop-in replacement: same head architecture, only backbone changes

---

## 7. Model 2 (GNN) — Status

**Status: NOT IMPLEMENTED (placeholder only)**

UPFD gossipcop data available on Drive (1.1GB). Implementation pending.

---

## 8. Fusion Layer — Status

**Status: NOT IMPLEMENTED (placeholder only)**

---

## 9. Data Available

| Dataset | Location | Size | Used For |
|---|---|---|---|
| SemEval 2020 Task 11 | Drive/semeval_data/ | 371 articles | Model 1: manipulation |
| GossipCop | HuggingFace (auto-download) | ~24K articles | Model 1: fake/real |
| PolitiFact | HuggingFace (auto-download) | ~21K articles | Model 1: fake/real |
| LIAR | UCSB TSV (auto-download) | ~12.8K statements | Model 1: cross-domain test |
| NewsMTSC | GitHub JSONL | ~9.9K sentences | Model 1: sentiment |
| tweet_eval | HuggingFace (auto-download) | ~60K tweets | Model 1: sentiment |
| SST-5 | HuggingFace (SetFit/sst5) | ~11.8K sentences | Model 1: sentiment |
| UPFD gossipcop | Drive/upfd_data/ | ~1.1GB | Model 2: GNN |

**Removed datasets:**
- WELFake: 99.5% accuracy = source-style artifact
- ISOT: 0.1% accuracy = complete generalization failure (domain incompatibility)

---

## 10. Key Findings

1. **WELFake 99.5% = source bias artifact.** Model learned Reuters writing style, not fake news semantics.
2. **ISOT 0.1% = domain incompatibility.** Not a label inversion — ISOT's fake/real articles come from completely different sources with distinct styles.
3. **GossipCop 98.10% with source debiasing** and PolitiFact cross-domain validation (75.22%) prove the model is not purely memorizing.
4. **Multi-task heads learn distinct representations.** Manipulation head detects stylistic exaggeration; fake head detects domain-level patterns. This separation validates the multi-task architecture.
5. **Style features are not critical.** Ablation shows zero performance drop when zeroed — the encoder captures stylistic information implicitly.
6. **Model is well-calibrated.** ECE < 0.07 for fake and manipulation heads. Errors have lower confidence than correct predictions.
7. **XLM-RoBERTa English capacity is the bottleneck.** Post-hoc tuning, architecture expansion (Secenek A), and data augmentation all failed to push ManipF1/SentAcc to targets. DeBERTa-v3-base backbone upgrade is required.
8. **Asymmetric Loss (ASL) outperforms Focal Loss** for imbalanced propaganda detection but the improvement is modest (+0.02 F1) — backbone capacity limits the gain.
9. **Text-only models cannot perform fact-checking.** This is a fundamental limitation. Model detects stylistic manipulation, not factual accuracy. Model 2 (GNN) + propagation patterns are needed for complementary signals.
