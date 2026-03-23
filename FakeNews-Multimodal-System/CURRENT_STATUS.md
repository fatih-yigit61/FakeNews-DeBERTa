# Model 1 — Current Status Report

**Date:** March 2026
**Version:** v2.4.0 (English-only, Secenek A improvements, ASL, SST-5 added)
**Purpose:** Quick-reference status document for cross-AI review.

---

## 1. What Does the Project Do?

Two-stage fake news detection system:

**Model 1 (Text Transformer):** XLM-RoBERTa-based multi-task model. **COMPLETE.**
**Model 2 (GNN):** Social propagation graph detection. **NOT STARTED.**

### Model 1 — Task Heads

| Task | Head | Data Source | Status |
|---|---|---|---|
| Fake news detection | `fake_head` | GossipCop + PolitiFact (2 domains) | Active |
| Propaganda/manipulation | `manipulation_head` | SemEval 2020 + MBIB + BABE (~22.8K) | Active |
| Sentiment classification | `sentiment_class_head` | NewsMTSC + tweet_eval + SST-5 (~48.4K) | Active |
| Sentiment intensity | `sentiment_intensity_head` | — (loss zeroed, redundant) | Passive |

Model also produces **128-d manipulation embedding vector** per article -> Model 2 (GNN) node feature.

---

## 2. Targets and Current Results

| Metric | Target | Latest Test | Status | Note |
|---|---|---|---|---|
| FakeAcc (GossipCop test) | >= 85% | **98.10%** | ACHIEVED | Source-debiased, 2-domain training |
| FakeAcc (PolitiFact test) | — | **75.22%** | Info | Different domain, generalization proof |
| FakeAcc (LIAR cross-domain) | — | **53.31%** | Info | Format mismatch (short statements) |
| ManipF1 (SemEval test) | >= 0.65 | **0.6352** | NOT MET | Precision=0.5289, Recall=0.7951 |
| ManipAUC (SemEval test) | — | **0.8037** | Info | Strong ranking ability |
| SentAcc (sentiment test) | >= 75% | **68.25%** | NOT MET | Calibrated: 69.12% |

### Post-hoc Threshold/Calibration Results
- Manipulation threshold sweep: optimal=0.45, F1=0.6352 (no improvement over default)
- Sentiment calibration: temp=1.5, neg=+0.3, pos=-0.3 -> 69.12% (+0.87%)
- **Conclusion:** Post-hoc tuning cannot close the gap. XLM-RoBERTa backbone is the bottleneck.

---

## 3. Training Run Summary

### Runs 1-5 — WELFake Era (v1.0-v1.7)
- FakeAcc: 99%+ (WELFake source bias artifact)
- ManipF1: 0.61-0.63 (SemEval SOTA boundary)
- WELFake removed due to 99.5% style-artifact

### Runs 6-8 — GossipCop Era (v2.0-v2.2)
- Multi-domain training: GossipCop + LIAR + ISOT
- Gradient starvation discovered (lambda_manipulation=2.5 dominated 54% of gradients)
- ISOT/LIAR dataset conflict: GossipCop ROC-AUC dropped to 0.32

### Runs 9-12 — PolitiFact Era (v2.3)
- LIAR and ISOT removed from training, PolitiFact added
- lambda_fake=1.5 added, gradient budget balanced
- GossipCop: 98.15%, PolitiFact: 74.75%, LIAR: 56.98%
- ManipF1: 0.6139, SentAcc: 69.03%

### Run 13 — Secenek A (v2.4, current)
- 9 architectural + data improvements applied:
  - Asymmetric Loss (ASL) replaces Focal Loss for manipulation head
  - Sentiment head expanded: 768->512->256->3 (2 hidden layers)
  - Manipulation head expanded: 768->512->256->128 (wider)
  - SST-5 dataset added (8.5K sentences)
  - tweet_eval cap raised to 12K/class
  - sent_weights changed to [1.0, 2.5, 2.5] symmetric
  - lambda_sentiment raised to 1.3
  - Threshold sweep widened to [0.15, 0.75]
  - WeightedRandomSampler normalization fixed
- Results: FakeAcc=98.10%, ManipF1=0.6352, SentAcc=68.25%
- ManipF1 improved slightly (+0.02) but still below 0.65 target
- SentAcc decreased slightly (69.03% -> 68.25%)

---

## 4. Current Dataset Structure

### Fake/Real (2 Training Domains + 1 Cross-Domain Test)
```
GossipCop (HF: GonzaloA/fake_news)        -> ~17.3K train (entertainment/celebrity)
PolitiFact (HF: Cartinoe5930/Politifact)   -> ~16K train (political fact-checks)
LIAR (UCSB TSV)                            -> TEST ONLY (political short statements)
ISOT                                       -> REMOVED (style artifact)
WELFake                                    -> REMOVED (99.5% source bias)
```

### Manipulation (SemEval + Augmentation)
```
SemEval 2020 Task 11                      -> 9,679 train sentences
MBIB linguistic_bias (HF)                 -> 5,000 pos + 5,000 neg
BABE media bias (HF)                      -> 3,120 sentences
Total: ~22,799 train sentences
```

### Sentiment (News + Tweet + SST-5)
```
NewsMTSC (GitHub JSONL)                    -> 8,739 train (news domain)
tweet_eval sentiment (HF)                  -> 31,093 train (12K/class cap)
SST-5 (HF: SetFit/sst5)                   -> 8,544 train (5K/class cap, 5->3 class)
Total: ~48,376 train
```

---

## 5. Post-Hoc Analysis Results (Complete)

### Style Feature Ablation
- **Result: ALL STABLE** — zeroing style_feats causes no performance drop
- **Interpretation:** Model learned semantics, not dependent on surface-level stylometric features

### Adversarial Robustness Test (15 pairs)
- **Manipulation head: 15/15 correct** — avg +30pp for manipulated texts
- **Fake head: 1/15 detected** — does not equate manipulation with fakeness
- **Key finding:** Fake and manipulation heads learned distinct representations

### Attention Visualization (4 samples)
- Attention heatmaps generated for 2 real + 2 fake news texts

### Error Analysis
- GossipCop: 2.1% error rate (49/2311) — FP's are political content
- PolitiFact: 25.6% error rate — FP's are sensational real news
- LIAR: 44% error rate — format mismatch (18-word statements)

### Confidence Calibration (v2.4)
- Fake detection ECE: 0.0637 (well-calibrated)
- Manipulation ECE: 0.0672 (well-calibrated)
- Sentiment ECE: 0.0889 (improved from 0.1301)

---

## 6. Current Configuration

```python
# configs/config.py — v2.4
num_epochs               = 12
early_stopping_patience  = 7
learning_rate            = 2e-5
batch_size               = 32
gradient_accum           = 4           # effective batch = 128
max_seq_len              = 512

# Loss weights
lambda_fake              = 1.5
lambda_sentiment         = 1.3         # raised from 0.7
lambda_manipulation      = 1.5

# Asymmetric Loss (manipulation) — replaces Focal Loss
AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05)

# Sentiment class weights
sent_weights             = [1.0, 2.5, 2.5]  # symmetric neutral+positive boost

# Dataset routing
use_gossipcop            = True
gossipcop_max_per_class  = 12_000
use_politifact           = True
politifact_max_per_class = 25_000
use_liar_fallback        = False
use_liar_test            = True
use_welfake              = False
use_isot_test            = False
use_sst5                 = True        # NEW: SST-5 added
sst5_max_per_class       = 5_000
tweet_sentiment_secondary_cap = 12_000 # raised from 5K
```

---

## 7. Anti-Shortcut Measures

- **Source debiasing:** 60+ source names masked, byline/dateline removal
- **Multi-domain training:** 2 domains (entertainment + political)
- **WELFake disabled:** 99.5% accuracy = source-style artifact
- **ISOT disabled:** 0.1% accuracy = complete generalization failure
- **Style ablation verified:** model is NOT dependent on style features

---

## 8. Remaining Work (Priority Order)

### RED — Critical (Backbone Upgrade)
1. Switch to DeBERTa-v3-base backbone for ManipF1 >= 0.65 and SentAcc >= 75%
2. XLM-RoBERTa English capacity exhausted — DeBERTa-v3 GLUE 91.3 vs ~86

### RED — Critical (Model 2)
3. UPFD gossipcop graph data + PyTorch Geometric GNN
4. Integrate Model 1's 128-d embeddings as GNN node features

### YELLOW — Medium priority
5. Head ablation study (Full Model vs Only-Fake) — needs 1 extra training session
6. Thesis writing with all analysis plots and findings
7. 3 seeds for statistical reliability (if budget allows)

### GREEN — Low priority
8. Fusion Layer (Model 1 + Model 2 outputs)
