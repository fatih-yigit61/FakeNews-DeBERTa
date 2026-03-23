import json
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from configs.config import TrainerConfig
from src.features.stylometry import StyleScaler, StylometricExtractor
from src.models.text_transformer import OptimizedMultiTaskModel
from src.preprocessing.data_loader import PropagandaDataset, SemEvalParser, SentenceSample, SentimentDataset, SimpleNewsDataset
from src.training.loss import MultiTaskLoss
from src.utils.common import get_device, set_seed


class _LoadedSplit(NamedTuple):
    """Return type for each dataset loader helper."""
    train_ds: Dataset
    val_ds: Dataset
    train_w: List[float]   # per-sample weights for WeightedRandomSampler
    test_ds: Optional[Dataset] = None
    test_name: Optional[str] = None   # key for self.test_loaders


class Model1ExpertTrainer:
    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.cfg = config or TrainerConfig()
        self.device = get_device()
        set_seed(self.cfg.seed)
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.gnn_output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model: Optional[OptimizedMultiTaskModel] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loaders: Dict[str, DataLoader] = {}
        self.best_f1 = 0.0
        self._manip_threshold: float = 0.5   # updated by threshold sweep in _evaluate()
        self._style_scaler: Optional[StyleScaler] = StyleScaler() if self.cfg.normalize_style else None

    # ─────────────────────────────────────────────────────────────────────────
    # Private dataset loader helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_semeval(self) -> _LoadedSplit:
        """Parse SemEval 2020 Task 11 and do a 3-way article-ID split.

        Uses train-articles/ for training.  dev-articles/ (when labels exist) is
        registered as a separate test loader — it was never seen during training.
        The 15% article-ID validation split comes from train-articles/ only.
        """
        # Diagnostic: show the resolved path and file count before parsing
        articles_path = Path(self.cfg.articles_dir)
        labels_path   = Path(self.cfg.labels_dir)
        n_art_files = len(list(articles_path.glob("article*.txt"))) if articles_path.exists() else -1
        print(f"SemEval articles_dir : {articles_path.resolve()}  (exists={articles_path.exists()}, article*.txt={n_art_files})")
        print(f"SemEval labels_dir   : {labels_path.resolve()}  (exists={labels_path.exists()})")

        parser = SemEvalParser(self.cfg.articles_dir, self.cfg.labels_dir)
        semeval_samples = parser.parse()

        if not semeval_samples:
            print("[warn] SemEval returned 0 samples — check that train-articles/ symlink points to the "
                  "correct Drive folder and that article*.txt files exist inside it. "
                  "Skipping SemEval; manipulation_head will not train.")
            # Return a minimal split so load_data() can continue without SemEval
            # (fake_head and sentiment_head will still train from WELFake / tweet_eval)
            return None

        article_ids = sorted({s.article_id for s in semeval_samples})
        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(article_ids)

        # 3-way split on article IDs so the same article never appears in two splits
        n_total = len(article_ids)
        n_val  = int(n_total * self.cfg.val_split)
        n_test = int(n_total * self.cfg.val_split)   # same fraction as val for test
        val_ids  = set(article_ids[:n_val])
        test_ids = set(article_ids[n_val : n_val + n_test])

        train_s = [s for s in semeval_samples if s.article_id not in val_ids and s.article_id not in test_ids]
        val_s   = [s for s in semeval_samples if s.article_id in val_ids]
        test_s  = [s for s in semeval_samples if s.article_id in test_ids]

        n_semeval_train = len(train_s)
        print(f"SemEval: {n_semeval_train} train / {len(val_s)} val / {len(test_s)} test sentences "
              f"({sum(s.label for s in train_s)} positive)")

        # ── Augmentation: MBIB + BABE (train only, val/test stay pure SemEval) ──
        aug_samples: list = []
        if not getattr(self.cfg, "use_manip_augmentation", True):
            print("Manipulation augmentation disabled via config")
        else:
            try:
                from datasets import load_dataset as hf_load

                # MBIB linguistic_bias: sentence-level linguistic manipulation (401K total, cap 5K/class)
                _mbib = hf_load("mediabiasgroup/mbib-base", split="linguistic_bias")
                _mbib_texts = set()
                _mbib_pos, _mbib_neg = 0, 0
                _cap = self.cfg.manip_aug_cap_per_class
                for row in _mbib:
                    t = row["text"]
                    lab = int(row["label"])
                    if not isinstance(t, str) or len(t.strip()) < 10:
                        continue
                    if lab == 1 and _mbib_pos >= _cap:
                        continue
                    if lab == 0 and _mbib_neg >= _cap:
                        continue
                    if t.strip() in _mbib_texts:
                        continue
                    _mbib_texts.add(t.strip())
                    aug_samples.append(SentenceSample(
                        article_id=f"mbib_{len(aug_samples)}", sentence_id=0,
                        text=t.strip(), label=lab, char_start=0, char_end=len(t.strip())
                    ))
                    if lab == 1:
                        _mbib_pos += 1
                    else:
                        _mbib_neg += 1
                    if _mbib_pos >= _cap and _mbib_neg >= _cap:
                        break
                print(f"MBIB linguistic_bias augmentation: {_mbib_pos} pos + {_mbib_neg} neg = {_mbib_pos + _mbib_neg}")

                # BABE: media bias sentence-level (3.1K total)
                _babe = hf_load("mediabiasgroup/BABE", split="train")
                _babe_count = 0
                for row in _babe:
                    t = row["text"]
                    lab = int(row["label"])
                    if isinstance(t, str) and len(t.strip()) >= 10 and t.strip() not in _mbib_texts:
                        aug_samples.append(SentenceSample(
                            article_id=f"babe_{_babe_count}", sentence_id=0,
                            text=t.strip(), label=lab, char_start=0, char_end=len(t.strip())
                        ))
                        _babe_count += 1
                print(f"BABE augmentation: {_babe_count} sentences")
            except Exception as e:
                print(f"[warn] Manipulation augmentation failed: {e}")

        if aug_samples:
            rng.shuffle(aug_samples)
            train_s = train_s + aug_samples
            print(f"Manipulation train total: {n_semeval_train} SemEval + {len(aug_samples)} augmented = {len(train_s)}")

        m_pos = sum(s.label for s in train_s)
        m_neg = len(train_s) - m_pos
        train_w = [1.0 / max(m_neg, 1) if s.label == 0 else 1.0 / max(m_pos, 1) for s in train_s]

        # dev-articles/ is a separate, cleaner held-out set if labels are available
        dev_articles_dir = Path(self.cfg.articles_dir).parent / "dev-articles"
        dev_labels_dir   = Path(self.cfg.labels_dir).parent / "dev-labels-task2-technique-classification"
        dev_ds: Optional[Dataset] = None
        if dev_articles_dir.exists() and dev_labels_dir.exists():
            dev_parser = SemEvalParser(str(dev_articles_dir), str(dev_labels_dir))
            dev_samples = dev_parser.parse()
            if dev_samples:
                dev_ds = PropagandaDataset(dev_samples, self.tokenizer, self.cfg.max_seq_len, self._style_scaler)
                print(f"SemEval dev-articles: {len(dev_samples)} sentences loaded as held-out test")

        # Fall back to the 15% article-ID test split when dev-articles labels are absent
        test_name = "semeval_dev" if dev_ds is not None else "semeval_test"
        chosen_test_ds = dev_ds if dev_ds is not None else PropagandaDataset(
            test_s, self.tokenizer, self.cfg.max_seq_len, self._style_scaler
        )

        return _LoadedSplit(
            train_ds  = PropagandaDataset(train_s, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds    = PropagandaDataset(val_s,   self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w   = train_w,
            test_ds   = chosen_test_ds,
            test_name = test_name,
        )

    def _load_welfake(self) -> Optional[_LoadedSplit]:
        """Load WELFake CSV with dedup, style normalization, and topic-aware split.

        Key improvements over v1.x:
        1. Title removed from input (cfg.use_title_in_fake=False) to prevent
           stylistic shortcut learning from sensational headlines.
        2. Exact-duplicate removal on text body.
        3. Style normalization (collapse ALL CAPS, repeated punctuation) to reduce
           surface-level cues that leak source identity rather than veracity.
        4. Topic-aware train/val/test splitting via TF-IDF + KMeans clustering
           so articles about the same event don't appear in multiple splits.
        """
        wf_path = Path(self.cfg.welfake_csv)
        if not wf_path.exists():
            print(f"[warn] WELFake CSV not found at {wf_path} — fake_head will not train")
            return None

        import pandas as pd
        from src.preprocessing.text_cleaner import normalize_style

        df = pd.read_csv(wf_path)
        text_col  = "text"  if "text"  in df.columns else df.columns[1]
        label_col = "label" if "label" in df.columns else df.columns[-1]

        # ── Build text column (title removed by default) ──────────────────────
        if self.cfg.use_title_in_fake:
            title_col = "title" if "title" in df.columns else None
            if title_col:
                df["_combined"] = df[title_col].astype(str) + " " + df[text_col].astype(str)
            else:
                df["_combined"] = df[text_col].astype(str)
        else:
            df["_combined"] = df[text_col].astype(str)

        df["_label"] = df[label_col].astype(int)
        df = df[["_combined", "_label"]].dropna()

        # ── Deduplicate ───────────────────────────────────────────────────────
        n_before = len(df)
        df = df.drop_duplicates(subset=["_combined"], keep="first")
        n_dupes = n_before - len(df)
        print(f"WELFake dedup: removed {n_dupes} exact duplicates ({n_before} → {len(df)})")

        # ── Style normalization ───────────────────────────────────────────────
        if self.cfg.normalize_welfake_style:
            df["_combined"] = df["_combined"].apply(normalize_style)

        # ── Balanced per-class cap ────────────────────────────────────────────
        cap = self.cfg.welfake_max_per_class
        df = (df.groupby("_label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), cap), random_state=self.cfg.seed))
                .reset_index(drop=True))

        texts  = df["_combined"].tolist()
        labels = df["_label"].tolist()

        # ── Topic-aware split via TF-IDF + KMeans ─────────────────────────────
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            self._topic_aware_split(texts, labels)

        f_pos = sum(train_labels)
        f_neg = len(train_labels) - f_pos
        train_w = [1.0 / max(f_neg, 1) if l == 0 else 1.0 / max(f_pos, 1) for l in train_labels]

        print(f"WELFake: {len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test "
              f"({sum(train_labels)} fake / {f_neg} real in train) [topic-aware split, "
              f"title={'included' if self.cfg.use_title_in_fake else 'excluded'}]")

        return _LoadedSplit(
            train_ds  = SimpleNewsDataset(train_texts, train_labels, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds    = SimpleNewsDataset(val_texts,   val_labels,   self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w   = train_w,
            test_ds   = SimpleNewsDataset(test_texts, test_labels, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name = "welfake_test",
        )

    def _topic_aware_split(
        self,
        texts: List[str],
        labels: List[int],
    ):
        """Split data by topic clusters so same-topic articles stay in one split.

        Uses TF-IDF + MiniBatchKMeans to assign each text to a topic cluster,
        then distributes entire clusters to train/val/test splits.  This prevents
        topic-level leakage where the model memorises event-specific vocabulary
        rather than learning veracity signals.
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        n_clusters = self.cfg.welfake_topic_clusters
        rng = np.random.default_rng(self.cfg.seed)

        # TF-IDF on truncated text (first 500 chars for speed)
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                                max_df=0.95, min_df=2)
        X = tfidf.fit_transform([t[:500] for t in texts])
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.cfg.seed,
                             batch_size=1024, n_init=3)
        clusters = km.fit_predict(X)

        # Distribute clusters to splits
        unique_clusters = list(set(clusters))
        rng.shuffle(unique_clusters)
        n_test = max(1, int(len(unique_clusters) * self.cfg.welfake_test_split))
        n_val  = max(1, int(len(unique_clusters) * self.cfg.val_split))
        test_cls = set(unique_clusters[:n_test])
        val_cls  = set(unique_clusters[n_test:n_test + n_val])

        train_t, val_t, test_t = [], [], []
        train_l, val_l, test_l = [], [], []
        for t, l, c in zip(texts, labels, clusters):
            if c in test_cls:
                test_t.append(t); test_l.append(l)
            elif c in val_cls:
                val_t.append(t); val_l.append(l)
            else:
                train_t.append(t); train_l.append(l)

        print(f"  Topic-aware split: {n_clusters} clusters → "
              f"{len(unique_clusters) - n_test - n_val} train / {n_val} val / {n_test} test clusters")

        return train_t, val_t, test_t, train_l, val_l, test_l

    def _load_sentiment_data(self) -> Optional[_LoadedSplit]:
        """Load combined sentiment data: NewsMTSC (primary) + tweet_eval (secondary).

        NewsMTSC (~11K news sentences, sentence-level sentiment):
          - Best domain match for news-focused sentiment analysis
          - Labels mapped to: 0=negative, 1=neutral, 2=positive
          - Uses HF native splits if available, else manual 70/15/15

        tweet_eval (secondary, 3,000/class):
          - Kept for linguistic volume and diversity
          - Lower cap since NewsMTSC is the primary source

        financial_phrasebank: DISABLED (finance domain doesn't generalize to general news)

        Combined dataset is shuffled and re-weighted by class frequency for the sampler.
        """
        from collections import Counter

        rng = np.random.default_rng(self.cfg.seed)
        all_train_texts:  List[str] = []
        all_train_labels: List[int] = []
        all_val_texts:    List[str] = []
        all_val_labels:   List[int] = []
        all_test_texts:   List[str] = []
        all_test_labels:  List[int] = []

        # ── 1. NewsMTSC (primary, news-domain sentiment) ─────────────────────
        if self.cfg.use_newsmtsc:
            try:
                # NewsMTSC uses a custom HF script that newer datasets versions reject.
                # Bypass: download JSONL splits directly from the GitHub source.
                _NEWSMTSC_BASE = (
                    "https://raw.githubusercontent.com/fhamborg/NewsMTSC/"
                    "6b838e00f54423c253806327a0ae24dbffa24c9e/"
                    "NewsSentiment/experiments/default/datasets/newsmtsc-rw-hf/"
                )
                import json
                from urllib.request import urlopen

                def _download_jsonl(url: str):
                    texts, labels = [], []
                    _label_map = {-1: 0, 0: 1, 1: 2}
                    response = urlopen(url)
                    for line in response:
                        row = json.loads(line.decode("utf-8"))
                        t = row.get("sentence", "")
                        raw_label = row.get("polarity")
                        if isinstance(t, str) and len(t.strip()) >= 5 and raw_label is not None:
                            mapped = _label_map.get(raw_label, raw_label)
                            if mapped in (0, 1, 2):
                                texts.append(t.strip())
                                labels.append(mapped)
                    return texts, labels

                nm_tr_t, nm_tr_l = _download_jsonl(_NEWSMTSC_BASE + "train.jsonl")
                nm_va_t, nm_va_l = _download_jsonl(_NEWSMTSC_BASE + "dev.jsonl")
                nm_te_t, nm_te_l = _download_jsonl(_NEWSMTSC_BASE + "test.jsonl")

                all_train_texts.extend(nm_tr_t); all_train_labels.extend(nm_tr_l)
                all_val_texts.extend(nm_va_t);   all_val_labels.extend(nm_va_l)
                all_test_texts.extend(nm_te_t);  all_test_labels.extend(nm_te_l)

                print(f"NewsMTSC (direct JSONL): {len(nm_tr_t)} train / "
                      f"{len(nm_va_t)} val / {len(nm_te_t)} test "
                      f"(neg/neu/pos={nm_tr_l.count(0)}/{nm_tr_l.count(1)}/{nm_tr_l.count(2)})")
            except Exception as e:
                print(f"[warn] NewsMTSC load failed: {e} — trying fallback sources")

        # ── 1b. financial_phrasebank (legacy, disabled by default) ────────────
        if self.cfg.use_financial_phrasebank and not all_train_texts:
            _fb_candidates = [
                ("nickmuchi/financial-classification", "text", "label",
                 {"Negative": 0, "Neutral": 1, "Positive": 2, 0: 0, 1: 1, 2: 2}),
                ("zeroshot/twitter-financial-news-sentiment", "text", "label",
                 {"Bearish": 0, "Neutral": 1, "Bullish": 2, 0: 0, 1: 1, 2: 2}),
            ]
            for _hf_id, _tc, _lc, _lmap in _fb_candidates:
                try:
                    from datasets import load_dataset as hf_load
                    _fb = hf_load(_hf_id)
                    _split_key = "train" if "train" in _fb else list(_fb.keys())[0]
                    _fb_split  = _fb[_split_key]
                    _fb_texts  = list(_fb_split[_tc])
                    _fb_raw    = list(_fb_split[_lc])
                    _fb_labels = [_lmap.get(l, int(l)) for l in _fb_raw]

                    idx = list(range(len(_fb_texts)))
                    rng.shuffle(idx)
                    n       = len(idx)
                    n_val   = int(n * 0.15)
                    n_test  = int(n * 0.15)
                    val_idx   = idx[:n_val]
                    test_idx  = idx[n_val:n_val + n_test]
                    train_idx = idx[n_val + n_test:]

                    all_train_texts.extend([_fb_texts[i]  for i in train_idx])
                    all_train_labels.extend([_fb_labels[i] for i in train_idx])
                    all_val_texts.extend([_fb_texts[i]    for i in val_idx])
                    all_val_labels.extend([_fb_labels[i]  for i in val_idx])
                    all_test_texts.extend([_fb_texts[i]   for i in test_idx])
                    all_test_labels.extend([_fb_labels[i] for i in test_idx])

                    print(f"news-sentiment fallback ({_hf_id}): {len(train_idx)} train / "
                          f"{len(val_idx)} val / {len(test_idx)} test")
                    break
                except Exception as e:
                    print(f"[warn] news-sentiment {_hf_id} load failed: {e}")
            else:
                print("[warn] All news-domain sentiment sources failed — using tweet_eval only")

        # ── 2. tweet_eval (secondary, capped for volume) ─────────────────────
        if self.cfg.use_tweet_sentiment:
            try:
                from datasets import load_dataset as hf_load
                ds = hf_load("tweet_eval", "sentiment")
            except Exception as e:
                print(f"[warn] tweet_eval load failed: {e}")
                ds = None

            if ds is not None:
                # Use higher cap when news-domain source failed (primary unavailable)
                cap = self.cfg.tweet_sentiment_secondary_cap if all_train_texts else self.cfg.tweet_sentiment_max_per_class

                def _sample_balanced(split_name: str):
                    split     = ds[split_name]
                    texts_all  = split["text"]
                    labels_all = split["label"]
                    chosen_idx: List[int] = []
                    for cls in range(3):
                        cls_idx = [i for i, lb in enumerate(labels_all) if lb == cls]
                        chosen  = rng.choice(cls_idx, size=min(len(cls_idx), cap), replace=False).tolist()
                        chosen_idx.extend(chosen)
                    rng.shuffle(chosen_idx)
                    return [texts_all[i] for i in chosen_idx], [labels_all[i] for i in chosen_idx]

                tw_tr_t, tw_tr_l = _sample_balanced("train")
                tw_va_t, tw_va_l = _sample_balanced("validation")
                tw_te_t, tw_te_l = _sample_balanced("test")

                all_train_texts.extend(tw_tr_t);  all_train_labels.extend(tw_tr_l)
                all_val_texts.extend(tw_va_t);    all_val_labels.extend(tw_va_l)
                all_test_texts.extend(tw_te_t);   all_test_labels.extend(tw_te_l)

                print(f"tweet_eval (secondary, {cap}/class): {len(tw_tr_t)} train / "
                      f"{len(tw_va_t)} val / {len(tw_te_t)} test "
                      f"(neg/neu/pos={tw_tr_l.count(0)}/{tw_tr_l.count(1)}/{tw_tr_l.count(2)})")

        # ── 3. SST-5 (Stanford Sentiment Treebank, 5-class → 3-class) ──────────
        if self.cfg.use_sst5:
            try:
                from datasets import load_dataset as hf_load
                sst5_ds = hf_load("SetFit/sst5")
                # SST-5 labels: 0=very_neg, 1=neg, 2=neutral, 3=pos, 4=very_pos
                # Map to 3-class: {0,1}→0(neg), {2}→1(neutral), {3,4}→2(pos)
                _sst5_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

                def _load_sst5_split(split_name: str, cap: int):
                    split = sst5_ds[split_name]
                    texts = list(split["text"])
                    labels = [_sst5_map[l] for l in split["label"]]
                    # Per-class cap
                    from collections import Counter as _C
                    cls_indices: Dict[int, List[int]] = {0: [], 1: [], 2: []}
                    for i, lb in enumerate(labels):
                        cls_indices[lb].append(i)
                    chosen: List[int] = []
                    for cls in range(3):
                        idx = cls_indices[cls]
                        if len(idx) > cap:
                            chosen.extend(rng.choice(idx, size=cap, replace=False).tolist())
                        else:
                            chosen.extend(idx)
                    rng.shuffle(chosen)
                    return [texts[i] for i in chosen], [labels[i] for i in chosen]

                sst_tr_t, sst_tr_l = _load_sst5_split("train", self.cfg.sst5_max_per_class)
                sst_va_t, sst_va_l = _load_sst5_split("validation", self.cfg.sst5_max_per_class)
                sst_te_t, sst_te_l = _load_sst5_split("test", self.cfg.sst5_max_per_class)

                all_train_texts.extend(sst_tr_t);  all_train_labels.extend(sst_tr_l)
                all_val_texts.extend(sst_va_t);    all_val_labels.extend(sst_va_l)
                all_test_texts.extend(sst_te_t);   all_test_labels.extend(sst_te_l)

                print(f"SST-5→3-class ({self.cfg.sst5_max_per_class}/class cap): "
                      f"{len(sst_tr_t)} train / {len(sst_va_t)} val / {len(sst_te_t)} test "
                      f"(neg/neu/pos={sst_tr_l.count(0)}/{sst_tr_l.count(1)}/{sst_tr_l.count(2)})")
            except Exception as e:
                print(f"[warn] SST-5 load failed: {e}")

        if not all_train_texts:
            print("[warn] No sentiment data loaded — sentiment head will not train")
            return None

        # ── Shuffle combined splits ───────────────────────────────────────────
        def _shuffle(texts: List[str], labels: List[int]):
            idx = list(range(len(texts)))
            rng.shuffle(idx)
            return [texts[i] for i in idx], [labels[i] for i in idx]

        all_train_texts, all_train_labels = _shuffle(all_train_texts, all_train_labels)
        all_val_texts,   all_val_labels   = _shuffle(all_val_texts,   all_val_labels)
        all_test_texts,  all_test_labels  = _shuffle(all_test_texts,  all_test_labels)

        sent_cls_counts = Counter(all_train_labels)
        train_w = [1.0 / sent_cls_counts[l] for l in all_train_labels]

        print(f"Sentiment combined: {len(all_train_texts)} train / {len(all_val_texts)} val / "
              f"{len(all_test_texts)} test "
              f"(neg/neu/pos={all_train_labels.count(0)}/{all_train_labels.count(1)}/{all_train_labels.count(2)})")

        return _LoadedSplit(
            train_ds  = SentimentDataset(all_train_texts,  all_train_labels,  self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds    = SentimentDataset(all_val_texts,    all_val_labels,    self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w   = train_w,
            test_ds   = SentimentDataset(all_test_texts,   all_test_labels,   self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name = "sentiment_test",
        )

    def _load_proppy(self) -> Optional[_LoadedSplit]:
        """Load external propaganda dataset to augment SemEval for manipulation_head.

        Tries multiple HuggingFace dataset IDs in order; skips silently if all fail.
        Label convention: 1 = propaganda/manipulation, 0 = not (same as SemEval).
        Sentences are derived from article text via simple sentence splitting if needed.

        Candidate datasets (tried in order):
          - newsmediabias/Proppy-Corpus-Sentence-Level : sentence-level propaganda labels
          - Babelscape/multinerd : not propaganda — skipped if wrong format
        """
        if not self.cfg.use_proppy:
            return None

        _candidates = [
            ("newsmediabias/Proppy-Corpus-Sentence-Level", "text", "label"),
            ("newsmediabias/proppy", "text", "label"),
            ("Aliyar/propaganda-detection", "text", "label"),
        ]

        from datasets import load_dataset as hf_load

        ds = None
        text_col = "text"
        label_col = "label"
        hf_id_used = ""

        for hf_id, tc, lc in _candidates:
            try:
                _ds = hf_load(hf_id)
                _split = _ds["train"] if "train" in _ds else _ds[list(_ds.keys())[0]]
                if tc not in _split.column_names or lc not in _split.column_names:
                    continue
                ds = _ds
                text_col, label_col, hf_id_used = tc, lc, hf_id
                break
            except Exception:
                continue

        if ds is None:
            # ── Fallback: dev-articles dizinini augmentation olarak kullan ──────
            dev_dir = Path(self.cfg.articles_dir).parent / "dev-articles"
            dev_labels_dir = Path(self.cfg.labels_dir).parent / "dev-task-TC-template.out"
            if dev_dir.exists() and len(list(dev_dir.glob("article*.txt"))) > 0:
                print(f"[info] Proppy HF failed — using dev-articles as augmentation ({len(list(dev_dir.glob('article*.txt')))} articles)")
                from src.preprocessing.data_loader import SemEvalParser
                dev_parser = SemEvalParser(str(dev_dir), str(Path(self.cfg.labels_dir)))
                try:
                    dev_samples = dev_parser.parse()
                    if dev_samples:
                        rng = np.random.default_rng(self.cfg.seed + 1)
                        idx = list(range(len(dev_samples)))
                        rng.shuffle(idx)
                        n = len(idx)
                        n_val = int(n * 0.15)
                        n_test = int(n * 0.15)
                        val_idx = idx[:n_val]
                        test_idx = idx[n_val:n_val + n_test]
                        train_idx = idx[n_val + n_test:]

                        m_pos = sum(1 for i in train_idx if dev_samples[i].label == 1)
                        m_neg = len(train_idx) - m_pos
                        train_w = [1.0 / max(m_neg, 1) if dev_samples[i].label == 0 else 1.0 / max(m_pos, 1) for i in train_idx]

                        print(f"Dev-articles augmentation: {len(train_idx)} train / {len(val_idx)} val / "
                              f"{len(test_idx)} test  ({m_pos} positive)")

                        return _LoadedSplit(
                            train_ds=PropagandaDataset([dev_samples[i] for i in train_idx], self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
                            val_ds=PropagandaDataset([dev_samples[i] for i in val_idx], self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
                            train_w=train_w,
                            test_ds=PropagandaDataset([dev_samples[i] for i in test_idx], self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
                            test_name="proppy_test",
                        )
                except Exception as e:
                    print(f"[warn] Dev-articles fallback failed: {e}")

            print("[warn] Proppy: no loadable HF propaganda source found — manipulation_head uses SemEval only")
            return None

        rng = np.random.default_rng(self.cfg.seed + 1)
        split_key = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_key]
        raw_texts  = list(split[text_col])
        raw_labels = [int(l) for l in split[label_col]]

        # Deduplicate and shuffle
        seen: set = set()
        texts, labels = [], []
        for t, l in zip(raw_texts, raw_labels):
            if t not in seen and isinstance(t, str) and len(t.strip()) >= 10:
                seen.add(t)
                texts.append(t)
                labels.append(l)

        idx = list(range(len(texts)))
        rng.shuffle(idx)
        n      = len(idx)
        n_val  = int(n * 0.15)
        n_test = int(n * 0.15)
        val_idx   = idx[:n_val]
        test_idx  = idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]

        m_pos = sum(labels[i] for i in train_idx)
        m_neg = len(train_idx) - m_pos
        train_w = [1.0 / max(m_neg, 1) if labels[i] == 0 else 1.0 / max(m_pos, 1) for i in train_idx]

        print(f"Proppy ({hf_id_used}): {len(train_idx)} train / {len(val_idx)} val / "
              f"{len(test_idx)} test  ({m_pos} positive)")

        from src.preprocessing.data_loader import PropagandaDataset, SentenceSample

        def _make_samples(idxs: List[int]) -> List:
            return [
                SentenceSample(
                    article_id=f"proppy_{i}",
                    sentence_id=i,
                    text=texts[i],
                    label=labels[i],
                )
                for i in idxs
            ]

        return _LoadedSplit(
            train_ds = PropagandaDataset(_make_samples(train_idx), self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds   = PropagandaDataset(_make_samples(val_idx),   self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w  = train_w,
            test_ds  = PropagandaDataset(_make_samples(test_idx),  self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name = "proppy_test",
        )

    def _load_gossipcop(self) -> Optional[_LoadedSplit]:
        """Load GossipCop fake news dataset from HuggingFace.

        Replaces WELFake as the primary fake/real training source.
        GossipCop is fact-checked by GossipCop.com.
        Also aligns with UPFD graph data for Model 2 integration.
        Label convention: 0=real, 1=fake.

        ANTI-SHORTCUT MEASURES:
        - Source debiasing: organization names, bylines, datelines removed
        - Style normalization: ALL CAPS, repeated punctuation reduced
        - These force the model to learn content/claim features, not source style.
        """
        from datasets import load_dataset as hf_load
        from collections import Counter
        from src.preprocessing.text_cleaner import clean_for_fake_detection

        _candidates = [
            ("GonzaloA/fake_news", "text", "label"),
            ("jannaiklaas/fakenews", "text", "label"),
            ("mohammadjavadpirhadi/fake-news-detection-dataset-English", "Text", "Label"),
        ]

        ds = None
        text_col = label_col = hf_id_used = ""

        for hf_id, tc, lc in _candidates:
            try:
                _ds = hf_load(hf_id)
                _split = _ds["train"] if "train" in _ds else _ds[list(_ds.keys())[0]]
                if tc not in _split.column_names or lc not in _split.column_names:
                    continue
                ds = _ds
                text_col, label_col, hf_id_used = tc, lc, hf_id
                break
            except Exception:
                continue

        if ds is None:
            print("[warn] GossipCop: no HF source found — trying LIAR fallback")
            return None

        rng = np.random.default_rng(self.cfg.seed)
        split_key = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_key]
        raw_texts = list(split[text_col])
        raw_labels = [int(l) for l in split[label_col]]

        # Deduplicate + source debias + style normalize
        seen: set = set()
        texts, labels = [], []
        for t, l in zip(raw_texts, raw_labels):
            if isinstance(t, str) and len(t.strip()) >= 20 and t not in seen:
                seen.add(t)
                if l in (0, 1):
                    # KEY: remove source-identifying markers before training
                    cleaned = clean_for_fake_detection(t.strip())
                    if len(cleaned) >= 20:
                        texts.append(cleaned)
                        labels.append(l)

        # Balanced cap per class (use gossipcop-specific cap if available)
        cap = getattr(self.cfg, "gossipcop_max_per_class", self.cfg.welfake_max_per_class)
        from collections import defaultdict
        by_class: dict = defaultdict(list)
        for i, l in enumerate(labels):
            by_class[l].append(i)
        kept_idx: List[int] = []
        for cls, idxs in by_class.items():
            chosen = rng.choice(idxs, size=min(len(idxs), cap), replace=False).tolist()
            kept_idx.extend(chosen)
        rng.shuffle(kept_idx)
        texts = [texts[i] for i in kept_idx]
        labels = [labels[i] for i in kept_idx]

        n = len(texts)
        n_test = int(n * self.cfg.welfake_test_split)
        n_val = int(n * self.cfg.val_split)
        idx = list(range(n))
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        n_fake_train = sum(labels[i] for i in train_idx)
        n_real_train = len(train_idx) - n_fake_train
        train_w = [1.0 / max(n_real_train, 1) if labels[i] == 0 else 1.0 / max(n_fake_train, 1) for i in train_idx]

        print(f"GossipCop ({hf_id_used}): {len(train_idx)} train / {len(val_idx)} val / "
              f"{len(test_idx)} test ({n_fake_train} fake / {n_real_train} real in train) "
              f"[dedup, cap={cap}/class, source-debiased]")

        return _LoadedSplit(
            train_ds=SimpleNewsDataset([texts[i] for i in train_idx], [labels[i] for i in train_idx],
                                       self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds=SimpleNewsDataset([texts[i] for i in val_idx], [labels[i] for i in val_idx],
                                     self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w=train_w,
            test_ds=SimpleNewsDataset([texts[i] for i in test_idx], [labels[i] for i in test_idx],
                                      self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name="gossipcop_test",
        )

    def _load_liar(self) -> Optional[_LoadedSplit]:
        """Load LIAR dataset (PolitiFact fact-checks) as ADDITIONAL fake/real training source.

        Downloads TSV files directly from UCSB source (bypasses broken HF script).
        6-class labels mapped to binary:
          fake(1): pants-fire, false, barely-true
          real(0): half-true, mostly-true, true
        ~12.8K statements.

        IMPORTANT: LIAR adds political domain diversity — GossipCop is entertainment only.
        Training on both prevents the model from overfitting to a single domain's style.
        Source debiasing applied for consistency.
        """
        _LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
        from src.preprocessing.text_cleaner import clean_for_fake_detection

        # Label name → binary mapping
        _label_map = {
            "pants-fire": 1, "false": 1, "barely-true": 1,
            "half-true": 0, "mostly-true": 0, "true": 0,
        }

        def _parse_tsv(content: str):
            texts, labels = [], []
            for line in content.strip().split("\n"):
                cols = line.split("\t")
                if len(cols) < 3:
                    continue
                label_str = cols[1].strip().lower()
                statement = cols[2].strip()
                if label_str in _label_map and len(statement) >= 10:
                    cleaned = clean_for_fake_detection(statement)
                    if len(cleaned) >= 10:
                        texts.append(cleaned)
                        labels.append(_label_map[label_str])
            return texts, labels

        try:
            import zipfile, io
            from urllib.request import urlopen
            data = urlopen(_LIAR_URL, timeout=30).read()
            z = zipfile.ZipFile(io.BytesIO(data))

            tr_t, tr_l = _parse_tsv(z.read("train.tsv").decode("utf-8"))
            va_t, va_l = _parse_tsv(z.read("valid.tsv").decode("utf-8"))
            te_t, te_l = _parse_tsv(z.read("test.tsv").decode("utf-8"))
        except Exception as e:
            print(f"[warn] LIAR dataset load failed: {e}")
            return None

        n_fake = sum(tr_l)
        n_real = len(tr_l) - n_fake
        train_w = [1.0 / max(n_real, 1) if l == 0 else 1.0 / max(n_fake, 1) for l in tr_l]

        print(f"LIAR (PolitiFact, direct TSV): {len(tr_t)} train / {len(va_t)} val / {len(te_t)} test "
              f"({n_fake} fake / {n_real} real in train) [source-debiased]")

        return _LoadedSplit(
            train_ds=SimpleNewsDataset(tr_t, tr_l, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds=SimpleNewsDataset(va_t, va_l, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w=train_w,
            test_ds=SimpleNewsDataset(te_t, te_l, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name="liar_test",
        )

    def _load_politifact(self) -> Optional[_LoadedSplit]:
        """Load PolitiFact fake news dataset from HuggingFace (FakeNewsNet framework).

        PolitiFact is the political counterpart of GossipCop in the FakeNewsNet
        framework (Shu et al.). Both datasets use fact-check-based labeling and
        contain full-length articles, making them highly compatible for joint training.

        GossipCop = entertainment domain, PolitiFact = political domain.
        Training on both gives the model cross-domain exposure without the
        format conflicts that LIAR (short statements) caused.

        Label convention: 0=real, 1=fake.
        Source debiasing applied for consistency.
        """
        from datasets import load_dataset as hf_load
        from collections import defaultdict
        from src.preprocessing.text_cleaner import clean_for_fake_detection

        # Candidates ordered by size: Cartinoe5930 (21K) >> LittleFish (483)
        # Cartinoe5930 labels are INVERTED: 0=fake(pants-fire), 1=real(mostly-true)
        # We detect and flip to match our convention: 0=real, 1=fake.
        _candidates = [
            ("Cartinoe5930/Politifact_fake_news", "news", "label", True),   # needs_flip=True
            ("LittleFish-Coder/Fake_News_PolitiFact", "text", "label", False),
        ]

        ds = None
        text_col = label_col = hf_id_used = ""
        needs_flip = False

        for hf_id, tc, lc, flip in _candidates:
            try:
                _ds = hf_load(hf_id)
                _split = _ds["train"] if "train" in _ds else _ds[list(_ds.keys())[0]]
                if tc not in _split.column_names or lc not in _split.column_names:
                    print(f"[PolitiFact] {hf_id}: columns {_split.column_names} — missing {tc}/{lc}")
                    continue
                ds = _ds
                text_col, label_col, hf_id_used = tc, lc, hf_id
                needs_flip = flip
                break
            except Exception as e:
                print(f"[PolitiFact] {hf_id} failed: {str(e)[:80]}")
                continue

        if ds is None:
            print("[warn] PolitiFact: no HF source found — skipping political domain")
            return None

        if needs_flip:
            print(f"[PolitiFact] {hf_id_used}: flipping labels (0=fake→1, 1=real→0) to match convention")

        rng = np.random.default_rng(self.cfg.seed + 11)

        # Merge all available splits
        all_texts, all_labels = [], []
        for split_key in ds:
            split = ds[split_key]
            for t, l in zip(split[text_col], split[label_col]):
                if isinstance(t, str) and len(t.strip()) >= 20:
                    lab = int(l)
                    if lab in (0, 1):
                        # Flip if source uses inverted convention
                        final_label = (1 - lab) if needs_flip else lab
                        cleaned = clean_for_fake_detection(t.strip())
                        if len(cleaned) >= 20:
                            all_texts.append(cleaned)
                            all_labels.append(final_label)

        if len(all_texts) < 100:
            print(f"[warn] PolitiFact: only {len(all_texts)} samples after cleaning — skipping")
            return None

        # Deduplicate
        seen: set = set()
        texts, labels = [], []
        for t, l in zip(all_texts, all_labels):
            if t not in seen:
                seen.add(t)
                texts.append(t)
                labels.append(l)

        # Cap per class (use politifact_max_per_class config)
        cap = self.cfg.politifact_max_per_class
        by_class: dict = defaultdict(list)
        for i, l in enumerate(labels):
            by_class[l].append(i)
        kept_idx: List[int] = []
        for cls, idxs in by_class.items():
            chosen = rng.choice(idxs, size=min(len(idxs), cap), replace=False).tolist()
            kept_idx.extend(chosen)
        rng.shuffle(kept_idx)
        texts = [texts[i] for i in kept_idx]
        labels = [labels[i] for i in kept_idx]

        # Train / val / test split (same ratios as GossipCop)
        n = len(texts)
        n_test = int(n * self.cfg.welfake_test_split)  # 10% test
        n_val = int(n * self.cfg.val_split)              # 15% val
        idx = list(range(n))
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        n_fake_train = sum(labels[i] for i in train_idx)
        n_real_train = len(train_idx) - n_fake_train
        train_w = [1.0 / max(n_real_train, 1) if labels[i] == 0
                   else 1.0 / max(n_fake_train, 1) for i in train_idx]

        print(f"PolitiFact ({hf_id_used}): {len(train_idx)} train / {len(val_idx)} val / "
              f"{len(test_idx)} test ({n_fake_train} fake / {n_real_train} real in train) "
              f"[dedup, cap={cap}/class, source-debiased]")

        return _LoadedSplit(
            train_ds=SimpleNewsDataset([texts[i] for i in train_idx], [labels[i] for i in train_idx],
                                       self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            val_ds=SimpleNewsDataset([texts[i] for i in val_idx], [labels[i] for i in val_idx],
                                     self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            train_w=train_w,
            test_ds=SimpleNewsDataset([texts[i] for i in test_idx], [labels[i] for i in test_idx],
                                      self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name="politifact_test",
        )

    def _load_isot(self) -> Optional[_LoadedSplit]:
        """Load ISOT Fake News — split between TRAINING and CROSS-DOMAIN TEST.

        ISOT provides political fake news (a domain absent from GossipCop).
        By adding a portion to training, the model sees diverse fake-news styles
        and cannot shortcut on GossipCop's entertainment-gossip patterns alone.

        Split strategy:
          - 'train' split → used for training (political domain exposure)
          - 'test' split  → held-out cross-domain evaluation (never trained on)

        This is the KEY anti-memorization measure: multi-domain training forces
        the model to learn content-level signals rather than source-specific style.

        Label convention: 0=real, 1=fake (verified via subject column heuristic).
        Source debiasing applied to both training and test portions.
        """
        if not self.cfg.use_isot_test:
            return None
        from src.preprocessing.text_cleaner import clean_for_fake_detection

        # Try multiple HF sources in order of preference
        _isot_sources = [
            "mohammadjavadpirhadi/fake-news-detection-dataset-English",
            self.cfg.isot_hf_path,
        ]

        ds = None
        for src in _isot_sources:
            try:
                from datasets import load_dataset as hf_load
                ds = hf_load(src)
                print(f"[ISOT] Loaded from: {src}")
                break
            except Exception as e:
                print(f"[ISOT] {src} failed: {str(e)[:80]}")
                continue

        if ds is None:
            print("[warn] ISOT: all sources failed — cross-domain test skipped")
            return None

        def _extract_isot(split_data):
            """Extract texts and labels from an ISOT split with debiasing."""
            col_names = split_data.column_names

            # Text column
            text_col = None
            for candidate in ("text", "article_text", "content", "body"):
                if candidate in col_names:
                    text_col = candidate
                    break
            if text_col is None:
                text_col = next((c for c in col_names if c not in (
                    "target", "label", "labels", "title", "subject", "date", "Unnamed: 0")), None)
            if text_col is None:
                return None, None, col_names

            # Label column
            label_col = None
            for candidate in ("label", "target", "labels", "fake"):
                if candidate in col_names:
                    label_col = candidate
                    break
            if label_col is None:
                return None, None, col_names

            raw_texts = split_data[text_col]
            raw_labels = [int(l) for l in split_data[label_col]]

            # Auto-detect label convention via subject column
            needs_flip = False
            if "subject" in col_names:
                subjects = split_data["subject"]
                real_subjects = {"politicsnews", "worldnews", "politics news", "world news"}
                label_0_real = sum(1 for s, l in zip(subjects, raw_labels)
                                   if l == 0 and isinstance(s, str) and s.strip().lower() in real_subjects)
                label_1_real = sum(1 for s, l in zip(subjects, raw_labels)
                                   if l == 1 and isinstance(s, str) and s.strip().lower() in real_subjects)
                if label_1_real > label_0_real:
                    needs_flip = True
                    print(f"[ISOT] Label flip: label=1 has more real-news subjects "
                          f"({label_1_real} vs {label_0_real}). Flipping to 0=real, 1=fake.")

            texts, labels = [], []
            for t, l in zip(raw_texts, raw_labels):
                if isinstance(t, str) and len(t.strip()) >= 20:
                    cleaned = clean_for_fake_detection(t.strip())
                    if len(cleaned) >= 20:
                        final_label = (1 - l) if needs_flip else l
                        if final_label in (0, 1):
                            texts.append(cleaned)
                            labels.append(final_label)
            return texts, labels, col_names

        # ── Extract from train split (for training) ───────────────────────
        train_texts, train_labels = None, None
        if "train" in ds:
            train_texts, train_labels, _ = _extract_isot(ds["train"])

        # ── Extract from test split (for cross-domain evaluation) ─────────
        test_split_key = "test"
        if test_split_key not in ds:
            for key in ("validation", "valid", "train"):
                if key in ds:
                    test_split_key = key
                    break

        test_texts, test_labels, col_names = _extract_isot(ds[test_split_key])
        if test_texts is None:
            print(f"[warn] ISOT: cannot find text/label columns in {col_names}")
            return None

        rng = np.random.default_rng(self.cfg.seed + 7)

        # ── Prepare training portion from ISOT train split ────────────────
        isot_train_cap = self.cfg.isot_train_cap  # cap per class for training
        tr_t, tr_l, va_t, va_l = [], [], [], []
        if train_texts and isot_train_cap > 0:
            from collections import defaultdict
            by_class: dict = defaultdict(list)
            for i, l in enumerate(train_labels):
                by_class[l].append(i)
            kept_idx: List[int] = []
            for cls, idxs in by_class.items():
                chosen = rng.choice(idxs, size=min(len(idxs), isot_train_cap), replace=False).tolist()
                kept_idx.extend(chosen)
            rng.shuffle(kept_idx)
            # 85/15 train/val split
            n_val = int(len(kept_idx) * 0.15)
            val_idx = kept_idx[:n_val]
            train_idx = kept_idx[n_val:]
            tr_t = [train_texts[i] for i in train_idx]
            tr_l = [train_labels[i] for i in train_idx]
            va_t = [train_texts[i] for i in val_idx]
            va_l = [train_labels[i] for i in val_idx]

        # ── Prepare test portion (pure cross-domain, never in training) ───
        max_n = self.cfg.isot_test_max
        if max_n and len(test_texts) > max_n:
            idx = rng.choice(len(test_texts), size=max_n, replace=False).tolist()
            test_texts = [test_texts[i] for i in idx]
            test_labels = [test_labels[i] for i in idx]

        n_fake_train = sum(tr_l) if tr_l else 0
        n_real_train = len(tr_l) - n_fake_train
        n_fake_test = sum(test_labels)
        n_real_test = len(test_labels) - n_fake_test

        if tr_t:
            train_w = [1.0 / max(n_real_train, 1) if l == 0
                       else 1.0 / max(n_fake_train, 1) for l in tr_l]
            print(f"ISOT train portion: {len(tr_t)} train / {len(va_t)} val "
                  f"({n_fake_train} fake / {n_real_train} real) [cap={isot_train_cap}/class, source-debiased]")
        else:
            train_w = []

        print(f"ISOT cross-domain test: {len(test_texts)} samples "
              f"({n_fake_test} fake / {n_real_test} real) — NEVER in training")

        return _LoadedSplit(
            train_ds  = SimpleNewsDataset(tr_t, tr_l, self.tokenizer, self.cfg.max_seq_len, self._style_scaler) if tr_t else None,
            val_ds    = SimpleNewsDataset(va_t, va_l, self.tokenizer, self.cfg.max_seq_len, self._style_scaler) if va_t else None,
            train_w   = train_w,
            test_ds   = SimpleNewsDataset(test_texts, test_labels, self.tokenizer, self.cfg.max_seq_len, self._style_scaler),
            test_name = "isot_cross_domain",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public data loading entry point
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_semeval_symlinks() -> None:
        """Auto-create SemEval symlinks from Google Drive if running on Colab.

        Resolves the recurring issue where git clone/pull breaks symlinks
        to the Drive folder containing SemEval data.
        """
        import glob as _glob
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        for sub in ("train-articles", "train-labels-task2-technique-classification"):
            link_path = data_dir / sub
            # Skip if already valid (exists and is readable)
            if link_path.exists() and (link_path.is_dir() or link_path.is_file()):
                continue
            # Remove broken symlink if present
            if link_path.is_symlink():
                link_path.unlink()
            # Search for SemEval data in common Drive locations
            candidates = _glob.glob(f"/content/drive/MyDrive/Thesis_Results/semeval_data*/{sub}")
            if candidates:
                os.symlink(candidates[0], str(link_path))
                print(f"[symlink] {link_path} → {candidates[0]}")
            # Also check direct Drive path without wildcard
            elif not candidates:
                direct = Path(f"/content/drive/MyDrive/Thesis_Results/semeval_data/{sub}")
                if direct.exists():
                    os.symlink(str(direct), str(link_path))
                    print(f"[symlink] {link_path} → {direct}")

    def load_data(self) -> None:
        self._ensure_semeval_symlinks()
        semeval_split   = self._load_semeval()      # returns None if articles not found
        proppy_split    = self._load_proppy()       # external propaganda augmentation

        # ── Fake/Real dataset routing ──────────────────────────────────────────
        # GossipCop (entertainment) + PolitiFact (political) for training.
        # Both from FakeNewsNet framework — compatible format, different domains.
        # LIAR excluded from training: short statements conflict with full articles.
        fake_split: Optional[_LoadedSplit] = None
        politifact_split: Optional[_LoadedSplit] = None
        liar_split: Optional[_LoadedSplit] = None
        if self.cfg.use_gossipcop:
            fake_split = self._load_gossipcop()
        if self.cfg.use_politifact:
            politifact_split = self._load_politifact()
        if self.cfg.use_liar_fallback:
            liar_split = self._load_liar()
        elif self.cfg.use_liar_test:
            # Load LIAR for cross-domain test only (not in training)
            liar_split = self._load_liar()
        # If GossipCop failed, use LIAR as primary (fallback)
        if fake_split is None and liar_split is not None and self.cfg.use_liar_fallback:
            fake_split = liar_split
            liar_split = None
        if fake_split is None and self.cfg.use_welfake:
            fake_split = self._load_welfake()
        if fake_split is None and politifact_split is None:
            print("[warn] No fake/real dataset loaded — fake_head will not train")

        sentiment_split = self._load_sentiment_data()
        isot_split      = self._load_isot()

        # Determine which LIAR split goes to training vs test-only
        liar_train_split = liar_split if self.cfg.use_liar_fallback else None

        # Only include splits that have actual training data
        # GossipCop + PolitiFact: multi-domain training (same framework, compatible)
        # LIAR excluded: test-only for cross-domain evaluation
        # ISOT: test-only when isot_train_cap=0
        active_splits = [s for s in [semeval_split, proppy_split, fake_split, politifact_split,
                                      liar_train_split, sentiment_split, isot_split]
                         if s is not None and s.train_ds is not None]

        if not active_splits:
            raise RuntimeError("No training data found. Check data/ symlinks with 'make check-data'.")

        # ── Fit StyleScaler on training texts only (test texts must never appear here)
        all_train_texts: List[str] = []
        for split in active_splits:
            ds = split.train_ds
            if hasattr(ds, "samples"):
                all_train_texts.extend(s.text for s in ds.samples)
            elif hasattr(ds, "texts"):
                all_train_texts.extend(ds.texts)

        if self._style_scaler is not None and all_train_texts:
            all_style = np.stack([StylometricExtractor.extract(t) for t in all_train_texts])
            self._style_scaler.fit(all_style)

        # ── Re-build datasets with fitted scaler (scaler was None during helper calls)
        # Helpers pass the scaler reference, but fit() is called here → the scaler
        # object is mutated in place, so all Dataset objects already share it.

        # ── Build DataLoaders ─────────────────────────────────────────────────
        train_parts = [s.train_ds for s in active_splits]
        val_parts   = [s.val_ds   for s in active_splits]
        # Normalize per-split weights before concatenation so each task gets
        # equal sampling probability regardless of dataset size.
        # Without this, larger datasets (fake: ~33K) dominate smaller ones
        # (sentiment: ~24K, manipulation: ~23K) in the sampler.
        normalized_w: List[float] = []
        for s in active_splits:
            w_arr = np.array(s.train_w, dtype=np.float64)
            w_sum = w_arr.sum()
            if w_sum > 0:
                w_arr = w_arr / w_sum  # normalize to sum=1 per split
            normalized_w.extend(w_arr.tolist())
        sample_w = normalized_w

        train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
        val_ds   = ConcatDataset(val_parts)   if len(val_parts)   > 1 else val_parts[0]

        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        num_workers = min(8, os.cpu_count() or 1)
        self.train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, sampler=sampler, num_workers=num_workers)
        self.val_loader   = DataLoader(val_ds,   batch_size=self.cfg.batch_size * 2, shuffle=False, num_workers=num_workers)

        # ── Register test loaders ─────────────────────────────────────────────
        for split in [semeval_split, proppy_split, fake_split, politifact_split, liar_split, sentiment_split, isot_split]:
            if split is not None and split.test_ds is not None and split.test_name is not None:
                self.test_loaders[split.test_name] = DataLoader(
                    split.test_ds,
                    batch_size=self.cfg.batch_size * 2,
                    shuffle=False,
                    num_workers=num_workers,
                )
        if self.test_loaders:
            print(f"Held-out test loaders: {list(self.test_loaders.keys())}")

    # ─────────────────────────────────────────────────────────────────────────
    # Model + optimizer
    # ─────────────────────────────────────────────────────────────────────────

    def build_model(self) -> None:
        self.model = OptimizedMultiTaskModel(
            model_name=self.cfg.model_name,
            dropout_rate=0.1,
            use_style_in_fake=self.cfg.use_style_in_fake,
        ).to(self.device)

    def _build_optimizer(self) -> torch.optim.AdamW:
        """AdamW with layer-wise LR decay when layer_lr_decay < 1.0."""
        if self.cfg.layer_lr_decay >= 1.0:
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
            )
        base_lr = self.cfg.learning_rate
        decay = self.cfg.layer_lr_decay
        encoder_layers = self.model.encoder.encoder.layer
        n_layers = len(encoder_layers)
        param_groups: List[Dict] = []
        param_groups.append({
            "params": list(self.model.encoder.embeddings.parameters()),
            "lr": base_lr * (decay ** n_layers),
        })
        for i, layer in enumerate(encoder_layers):
            param_groups.append({
                "params": list(layer.parameters()),
                "lr": base_lr * (decay ** (n_layers - i - 1)),
            })
        head_params = [p for n, p in self.model.named_parameters() if "encoder" not in n]
        param_groups.append({"params": head_params, "lr": base_lr})
        return torch.optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        if self.train_loader is None:
            self.load_data()
        if self.model is None:
            self.build_model()
        criterion = MultiTaskLoss(
            self.cfg.lambda_sentiment,
            self.cfg.lambda_manipulation,
            lambda_fake=self.cfg.lambda_fake,
            use_focal_loss=self.cfg.use_focal_loss,
        )
        optimizer = self._build_optimizer()
        total_steps = len(self.train_loader) * self.cfg.num_epochs // self.cfg.gradient_accum
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * self.cfg.warmup_ratio), total_steps)
        # BF16 autocast for DeBERTa-v3: same speed as FP16 but NaN-safe (FP32 exponent range)
        # GradScaler is NOT needed for BF16 — only required for FP16
        use_amp = self.cfg.bf16 or self.cfg.fp16
        amp_dtype = torch.bfloat16 if self.cfg.bf16 else torch.float16
        scaler = GradScaler("cuda", enabled=self.cfg.fp16)  # disabled for bf16
        history = []
        epochs_no_improve = 0
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}"), start=1):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                style = batch["style_feats"].to(self.device)
                targets = {
                    "fake_label": batch["fake_label"].to(self.device),
                    "sentiment_label": batch["sentiment_label"].to(self.device),
                    "sentiment_intensity": batch["sentiment_intensity"].to(self.device),
                    "manipulation_label": batch["manipulation_label"].to(self.device),
                }
                with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    predictions = self.model(input_ids, attn_mask, style)
                    losses = criterion(predictions, targets)
                    loss = losses["total_loss"] / self.cfg.gradient_accum
                if self.cfg.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % self.cfg.gradient_accum == 0:
                    if self.cfg.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    if self.cfg.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                train_loss += loss.item() * self.cfg.gradient_accum
            val_metrics = self._evaluate()

            # NOTE: Curriculum freeze REMOVED.
            # Previously froze fake_head at 98% accuracy, but this reinforced
            # shortcut learning — the model learned source style quickly, got
            # frozen, and never had to learn deeper content features.
            # With source debiasing + multi-domain training, the fake_head needs
            # continuous gradient to learn genuine content-level signals.

            active, total = 0.0, 0.0
            if val_metrics.get("manipulation_f1", 0.0) > 0.0:
                total += val_metrics.get("manipulation_f1", 0.0); active += 1
            if val_metrics.get("fake_acc", 0.0) > 0.0:
                total += val_metrics.get("fake_acc", 0.0); active += 1
            if val_metrics.get("sentiment_acc", 0.0) > 0.0:
                total += val_metrics.get("sentiment_acc", 0.0); active += 1
            composite = total / max(active, 1)
            if composite > self.best_f1:
                self.best_f1 = composite
                epochs_no_improve = 0
                self._save_checkpoint(epoch, composite)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.early_stopping_patience:
                    history.append({
                        "epoch": epoch,
                        "train_loss": train_loss / max(len(self.train_loader), 1),
                        "composite_score": composite,
                        **val_metrics,
                    })
                    break
            history.append({
                "epoch": epoch,
                "train_loss": train_loss / max(len(self.train_loader), 1),
                "composite_score": composite,
                **val_metrics,
            })
        with open(Path(self.cfg.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self._save_training_plots(history)

    @staticmethod
    def _save_training_plots(history: List[Dict]) -> None:
        """Generate and save training curve plots from the per-epoch history list.

        Saves:
          outputs/model1/plots/training_curves.png  — 6-panel combined figure
          outputs/model1/plots/loss.png
          outputs/model1/plots/manipulation.png
          outputs/model1/plots/fake.png
          outputs/model1/plots/sentiment.png
          outputs/model1/plots/composite.png
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend — safe on Colab/headless
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            print("[warn] matplotlib not installed — skipping training plots")
            return

        if not history:
            return

        # Determine output dir from first entry's absence of a path — derive from config
        # We save next to training_history.json; caller is responsible for the dir.
        # Resolve by checking if outputs/model1 exists relative to cwd.
        plots_dir = Path("outputs/model1/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        epochs = [e["epoch"] for e in history]

        def _get(key: str) -> List[float]:
            return [e.get(key, float("nan")) for e in history]

        # ── Data series ────────────────────────────────────────────────────
        train_loss   = _get("train_loss")
        val_loss     = _get("total_loss")
        composite    = _get("composite_score")

        manip_f1     = _get("manipulation_f1")
        manip_auc    = _get("manipulation_roc_auc")
        manip_prauc  = _get("manipulation_pr_auc")
        manip_prec   = _get("manipulation_precision")
        manip_rec    = _get("manipulation_recall")
        manip_thresh = _get("best_manip_threshold")

        fake_acc     = _get("fake_acc")
        fake_auc     = _get("fake_roc_auc")
        fake_f1mac   = _get("fake_f1_macro")
        fake_f1_fake = _get("fake_f1_fake")
        fake_f1_real = _get("fake_f1_real")

        sent_acc     = _get("sentiment_acc")
        sent_f1mac   = _get("sentiment_f1_macro")
        sent_kappa   = _get("sentiment_cohen_kappa")
        sent_f1_neg  = _get("sentiment_f1_negative")
        sent_f1_neu  = _get("sentiment_f1_neutral")
        sent_f1_pos  = _get("sentiment_f1_positive")

        STYLE = dict(linewidth=1.8, marker="o", markersize=4)
        GRID  = dict(linestyle="--", alpha=0.4)

        # ── Helper to mark best epoch ──────────────────────────────────────
        def mark_best(ax, values, epochs_list, color="gold"):
            valid = [(v, i) for i, v in enumerate(values) if not (v != v)]  # drop NaN
            if not valid:
                return
            best_val, best_i = max(valid)
            ax.axvline(epochs_list[best_i], color=color, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.annotate(f"{best_val:.3f}", xy=(epochs_list[best_i], best_val),
                        xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)

        # ══════════════════════════════════════════════════════════════════
        # 1) Combined 6-panel figure
        # ══════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(3, 2, figsize=(13, 13))
        fig.suptitle("Model 1 — Training Curves (Validation Set)", fontsize=13, fontweight="bold", y=0.98)

        # Panel 1 — Loss
        ax = axes[0, 0]
        ax.plot(epochs, train_loss, label="Train Loss", color="#E74C3C", **STYLE)
        ax.plot(epochs, val_loss,   label="Val Loss",   color="#3498DB", **STYLE)
        ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(**GRID)

        # Panel 2 — Manipulation
        ax = axes[0, 1]
        ax.plot(epochs, manip_f1,    label="F1",      color="#8E44AD", **STYLE)
        ax.plot(epochs, manip_auc,   label="ROC-AUC", color="#2ECC71", **STYLE)
        ax.plot(epochs, manip_prauc, label="PR-AUC",  color="#F39C12", **STYLE)
        mark_best(ax, manip_f1, epochs)
        ax.axhline(0.75, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Target F1=0.75")
        ax.set_title("Manipulation Detection"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(**GRID)

        # Panel 3 — Manipulation Precision / Recall / Threshold
        ax = axes[1, 0]
        ax.plot(epochs, manip_prec,   label="Precision",  color="#1ABC9C", **STYLE)
        ax.plot(epochs, manip_rec,    label="Recall",     color="#E67E22", **STYLE)
        ax2 = ax.twinx()
        ax2.plot(epochs, manip_thresh, label="Threshold", color="#95A5A6", linestyle="--", linewidth=1.4)
        ax2.set_ylabel("Threshold", color="#95A5A6", fontsize=8)
        ax2.set_ylim(0.2, 0.8)
        ax.set_title("Manipulation Precision / Recall / Threshold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
        ax.grid(**GRID)

        # Panel 4 — Fake/Real
        ax = axes[1, 1]
        ax.plot(epochs, fake_acc,    label="Accuracy",  color="#3498DB", **STYLE)
        ax.plot(epochs, fake_auc,    label="ROC-AUC",   color="#E74C3C", **STYLE)
        ax.plot(epochs, fake_f1mac,  label="F1-macro",  color="#F39C12", **STYLE)
        ax.plot(epochs, fake_f1_fake, label="F1-fake",  color="#9B59B6", linestyle="--", linewidth=1.2)
        ax.plot(epochs, fake_f1_real, label="F1-real",  color="#27AE60", linestyle="--", linewidth=1.2)
        ax.axhline(0.85, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Target Acc=0.85")
        ax.set_title("Fake/Real Classification"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(**GRID)

        # Panel 5 — Sentiment
        ax = axes[2, 0]
        ax.plot(epochs, sent_acc,    label="Accuracy",    color="#3498DB", **STYLE)
        ax.plot(epochs, sent_f1mac,  label="F1-macro",    color="#E74C3C", **STYLE)
        ax.plot(epochs, sent_kappa,  label="Cohen Kappa", color="#F39C12", **STYLE)
        ax.plot(epochs, sent_f1_neg, label="F1-neg", color="#8E44AD", linestyle="--", linewidth=1.2)
        ax.plot(epochs, sent_f1_neu, label="F1-neu", color="#1ABC9C", linestyle="--", linewidth=1.2)
        ax.plot(epochs, sent_f1_pos, label="F1-pos", color="#E67E22", linestyle="--", linewidth=1.2)
        ax.axhline(0.80, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Target Acc=0.80")
        ax.set_title("Sentiment Classification"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(**GRID)

        # Panel 6 — Composite Score (model selection criterion)
        ax = axes[2, 1]
        ax.plot(epochs, composite, label="Composite", color="#2C3E50", **STYLE)
        mark_best(ax, composite, epochs, color="gold")
        best_comp_epoch = epochs[composite.index(max(composite))] if composite else 0
        ax.axvline(best_comp_epoch, color="gold", linestyle=":", linewidth=1.5,
                   label=f"Best epoch={best_comp_epoch}")
        ax.set_title("Composite Score (Model Selection)"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Composite"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.grid(**GRID)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        combined_path = plots_dir / "training_curves.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] Saved combined figure → {combined_path}")

        # ══════════════════════════════════════════════════════════════════
        # 2) Individual plots (easier to embed in thesis)
        # ══════════════════════════════════════════════════════════════════
        _individual = [
            ("loss.png",         "Loss Curves",
             [(train_loss, "Train Loss", "#E74C3C"), (val_loss, "Val Loss", "#3498DB")],
             "Loss"),
            ("manipulation.png", "Manipulation Detection",
             [(manip_f1,    "F1",      "#8E44AD"),
              (manip_auc,   "ROC-AUC", "#2ECC71"),
              (manip_prauc, "PR-AUC",  "#F39C12"),
              (manip_prec,  "Precision","#1ABC9C"),
              (manip_rec,   "Recall",  "#E67E22")],
             "Score"),
            ("fake.png",         "Fake/Real Classification",
             [(fake_acc,    "Accuracy",  "#3498DB"),
              (fake_auc,    "ROC-AUC",  "#E74C3C"),
              (fake_f1mac,  "F1-macro", "#F39C12"),
              (fake_f1_fake,"F1-fake",  "#9B59B6"),
              (fake_f1_real,"F1-real",  "#27AE60")],
             "Score"),
            ("sentiment.png",    "Sentiment Classification",
             [(sent_acc,   "Accuracy",    "#3498DB"),
              (sent_f1mac, "F1-macro",    "#E74C3C"),
              (sent_kappa, "Cohen Kappa", "#F39C12"),
              (sent_f1_neg,"F1-negative", "#8E44AD"),
              (sent_f1_neu,"F1-neutral",  "#1ABC9C"),
              (sent_f1_pos,"F1-positive", "#E67E22")],
             "Score"),
            ("composite.png",    "Composite Score (Model Selection)",
             [(composite, "Composite", "#2C3E50")],
             "Composite"),
        ]

        for fname, title, series, ylabel in _individual:
            fig_s, ax_s = plt.subplots(figsize=(8, 4.5))
            for values, label, color in series:
                ax_s.plot(epochs, values, label=label, color=color, **STYLE)
            ax_s.set_title(title, fontweight="bold")
            ax_s.set_xlabel("Epoch"); ax_s.set_ylabel(ylabel)
            ax_s.legend(fontsize=9); ax_s.grid(**GRID)
            if ylabel == "Score":
                ax_s.set_ylim(0, 1.05)
            fig_s.tight_layout()
            path = plots_dir / fname
            fig_s.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig_s)
            print(f"[plots] Saved → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation kernel — reusable for val and test
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader: DataLoader,
        update_threshold: bool = False,
    ) -> Dict[str, float]:
        """Core evaluation over a single DataLoader.

        Parameters
        ----------
        loader          : DataLoader to evaluate.
        update_threshold: If True, sweeps the manipulation threshold on this
                          loader and writes the best value to self._manip_threshold.
                          Should be True ONLY for the validation set — using it on
                          a test set would leak the threshold and inflate numbers.
        """
        self.model.eval()
        criterion = MultiTaskLoss(
            self.cfg.lambda_sentiment, self.cfg.lambda_manipulation,
            lambda_fake=self.cfg.lambda_fake, use_focal_loss=self.cfg.use_focal_loss
        )
        all_manip_probs:  List[float] = []
        all_manip_labels: List[int]   = []
        all_fake_probs:   List[float] = []   # P(fake) from softmax — needed for ROC-AUC
        all_fake_preds:   List[int]   = []
        all_fake_labels:  List[int]   = []
        all_sent_preds:   List[int]   = []
        all_sent_labels:  List[int]   = []
        total_loss = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            style = batch["style_feats"].to(self.device)
            targets = {
                "fake_label":          batch["fake_label"].to(self.device),
                "sentiment_label":     batch["sentiment_label"].to(self.device),
                "sentiment_intensity": batch["sentiment_intensity"].to(self.device),
                "manipulation_label":  batch["manipulation_label"].to(self.device),
            }
            _use_amp = self.cfg.bf16 or self.cfg.fp16
            _amp_dt = torch.bfloat16 if self.cfg.bf16 else torch.float16
            with autocast("cuda", enabled=_use_amp, dtype=_amp_dt):
                predictions = self.model(input_ids, attn_mask, style)
                losses = criterion(predictions, targets)
            total_loss += losses["total_loss"].float().item()

            # ── Manipulation (binary) ──────────────────────────────────────
            manip_label_np = targets["manipulation_label"].cpu().numpy()
            manip_mask = manip_label_np != -1
            if manip_mask.any():
                manip_probs = torch.sigmoid(predictions["manipulation_logits"]).cpu().numpy()
                all_manip_probs.extend(manip_probs[manip_mask].tolist())
                all_manip_labels.extend(manip_label_np[manip_mask].astype(int).tolist())

            # ── Fake/Real (binary via 2-class softmax) ─────────────────────
            fake_label_np = targets["fake_label"].cpu().numpy()
            fake_mask = fake_label_np != -1
            if fake_mask.any():
                fake_softmax = torch.softmax(predictions["fake_logits"], dim=-1).cpu().numpy()
                fake_preds_np = fake_softmax.argmax(axis=-1)
                all_fake_preds.extend(fake_preds_np[fake_mask].tolist())
                all_fake_labels.extend(fake_label_np[fake_mask].tolist())
                all_fake_probs.extend(fake_softmax[fake_mask, 1].tolist())  # P(class=1=fake)

            # ── Sentiment (3-class) ────────────────────────────────────────
            sent_label_np = targets["sentiment_label"].cpu().numpy()
            sent_mask = sent_label_np != -1
            if sent_mask.any():
                sent_preds = predictions["sentiment_logits"].argmax(dim=-1).cpu().numpy()
                all_sent_preds.extend(sent_preds[sent_mask].tolist())
                all_sent_labels.extend(sent_label_np[sent_mask].tolist())

        # ── Manipulation threshold + metrics ───────────────────────────────
        best_manip_f1, best_thresh = 0.0, self._manip_threshold
        metrics: Dict[str, float] = {
            "total_loss": total_loss / max(len(loader), 1),
            "n_samples": len(all_manip_labels) + len(all_fake_labels) + len(all_sent_labels),
        }

        if all_manip_probs:
            probs_arr  = np.array(all_manip_probs)
            labels_arr = np.array(all_manip_labels)
            if update_threshold:
                # Wide sweep [0.15, 0.75] with 0.01 step: imbalanced data (~20% positive)
                # may have optimal threshold far from 0.50.
                for thr in np.arange(0.15, 0.75, 0.01):
                    preds_thr = (probs_arr >= thr).astype(int)
                    f1_thr = f1_score(labels_arr, preds_thr, zero_division=0)
                    if f1_thr > best_manip_f1:
                        best_manip_f1 = f1_thr
                        best_thresh   = float(thr)
                self._manip_threshold = best_thresh
            else:
                preds_arr = (probs_arr >= self._manip_threshold).astype(int)
                best_manip_f1 = f1_score(labels_arr, preds_arr, zero_division=0)
                best_thresh = self._manip_threshold

            preds_final = (probs_arr >= best_thresh).astype(int)
            cm_m = confusion_matrix(labels_arr, preds_final, labels=[0, 1])
            tn_m, fp_m, fn_m, tp_m = cm_m.ravel() if cm_m.shape == (2, 2) else (0, 0, 0, 0)
            metrics.update({
                "manipulation_f1":          best_manip_f1,
                "manipulation_precision":   precision_score(labels_arr, preds_final, zero_division=0),
                "manipulation_recall":      recall_score(labels_arr, preds_final, zero_division=0),
                "manipulation_roc_auc":     roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.0,
                "manipulation_pr_auc":      average_precision_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.0,
                "manipulation_tp":          int(tp_m),
                "manipulation_fp":          int(fp_m),
                "manipulation_tn":          int(tn_m),
                "manipulation_fn":          int(fn_m),
                "best_manip_threshold":     best_thresh,
                "n_manip_samples":          len(labels_arr),
            })

        if all_fake_labels:
            fa_labels = np.array(all_fake_labels, dtype=int)
            fa_preds  = np.array(all_fake_preds,  dtype=int)
            fa_probs  = np.array(all_fake_probs)
            cm_f = confusion_matrix(fa_labels, fa_preds, labels=[0, 1])
            tn_f, fp_f, fn_f, tp_f = cm_f.ravel() if cm_f.shape == (2, 2) else (0, 0, 0, 0)
            f1_per = f1_score(fa_labels, fa_preds, average=None, zero_division=0, labels=[0, 1])
            pr_per = precision_score(fa_labels, fa_preds, average=None, zero_division=0, labels=[0, 1])
            rc_per = recall_score(fa_labels, fa_preds, average=None, zero_division=0, labels=[0, 1])
            metrics.update({
                "fake_acc":               accuracy_score(fa_labels, fa_preds),
                "fake_f1_macro":          f1_score(fa_labels, fa_preds, average="macro",    zero_division=0),
                "fake_f1_weighted":       f1_score(fa_labels, fa_preds, average="weighted", zero_division=0),
                "fake_f1_real":           float(f1_per[0]),
                "fake_f1_fake":           float(f1_per[1]),
                "fake_precision_real":    float(pr_per[0]),
                "fake_precision_fake":    float(pr_per[1]),
                "fake_recall_real":       float(rc_per[0]),
                "fake_recall_fake":       float(rc_per[1]),
                "fake_roc_auc":           roc_auc_score(fa_labels, fa_probs) if len(np.unique(fa_labels)) > 1 else 0.0,
                "fake_pr_auc":            average_precision_score(fa_labels, fa_probs) if len(np.unique(fa_labels)) > 1 else 0.0,
                "fake_tp":                int(tp_f),
                "fake_fp":                int(fp_f),
                "fake_tn":                int(tn_f),
                "fake_fn":                int(fn_f),
                "n_fake_samples":         len(fa_labels),
            })

        if all_sent_labels:
            sl = np.array(all_sent_labels, dtype=int)
            sp = np.array(all_sent_preds,  dtype=int)
            classes = [0, 1, 2]
            f1_per_s  = f1_score(sl, sp, average=None, zero_division=0, labels=classes)
            pr_per_s  = precision_score(sl, sp, average=None, zero_division=0, labels=classes)
            rc_per_s  = recall_score(sl, sp, average=None, zero_division=0, labels=classes)
            metrics.update({
                "sentiment_acc":              accuracy_score(sl, sp),
                "sentiment_f1_macro":         f1_score(sl, sp, average="macro",    zero_division=0),
                "sentiment_f1_weighted":      f1_score(sl, sp, average="weighted", zero_division=0),
                "sentiment_f1_negative":      float(f1_per_s[0]),
                "sentiment_f1_neutral":       float(f1_per_s[1]),
                "sentiment_f1_positive":      float(f1_per_s[2]),
                "sentiment_precision_macro":  precision_score(sl, sp, average="macro", zero_division=0),
                "sentiment_recall_macro":     recall_score(sl, sp, average="macro",    zero_division=0),
                "sentiment_precision_neg":    float(pr_per_s[0]),
                "sentiment_precision_neu":    float(pr_per_s[1]),
                "sentiment_precision_pos":    float(pr_per_s[2]),
                "sentiment_recall_neg":       float(rc_per_s[0]),
                "sentiment_recall_neu":       float(rc_per_s[1]),
                "sentiment_recall_pos":       float(rc_per_s[2]),
                "sentiment_cohen_kappa":      cohen_kappa_score(sl, sp) if len(np.unique(sl)) > 1 else 0.0,
                "n_sentiment_samples":        len(sl),
            })

        return metrics

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set and update the manipulation threshold.

        Called once per training epoch.  The threshold sweep runs on val data only
        so it cannot leak test-set information.
        """
        return self._evaluate_loader(self.val_loader, update_threshold=True)

    @torch.no_grad()
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Run held-out test evaluation after training is complete.

        Uses self._manip_threshold calibrated on the val set — never re-sweeps on
        test data to avoid inflating manipulation F1 via threshold overfitting.

        Returns a dict: {dataset_name: {metric_name: value}}.
        The result is also written to outputs/model1/test_results.json.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call load_best_model() first.")
        if not self.test_loaders:
            print("[warn] No test loaders registered.  Call load_data() first.")
            return {}

        # Calibrate threshold on val set before running test evaluation
        val_metrics = self._evaluate_loader(self.val_loader, update_threshold=True)

        results: Dict[str, Dict[str, float]] = {"val": val_metrics}
        for name, loader in self.test_loaders.items():
            print(f"Evaluating: {name} ...")
            results[name] = self._evaluate_loader(loader, update_threshold=False)

        # Persist results
        out_path = Path(self.cfg.output_dir) / "test_results.json"
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        tmp_path.replace(out_path)   # atomic rename — safe on Colab disconnect

        self._print_test_results(results)
        print(f"\nSaved to {out_path}")
        return results

    @staticmethod
    def _print_test_results(results: Dict[str, Dict[str, float]]) -> None:
        """Pretty-print comprehensive test metrics to stdout."""
        SEP = "─" * 62
        print(f"\n{'═'*62}")
        print(f"  TEST RESULTS — Full Metric Report")
        print(f"{'═'*62}")

        for ds_name, m in results.items():
            print(f"\n  ┌─ {ds_name} {'─'*(55 - len(ds_name))}")

            # ── Manipulation metrics ───────────────────────────────────────
            if m.get("manipulation_f1", 0.0) > 0.0 or "manipulation_roc_auc" in m:
                print(f"  │  [Manipulation Detection]  n={m.get('n_manip_samples','?')}")
                print(f"  │    Threshold : {m.get('best_manip_threshold', 0.5):.2f}")
                print(f"  │    F1        : {m.get('manipulation_f1', 0):.4f}")
                print(f"  │    Precision : {m.get('manipulation_precision', 0):.4f}")
                print(f"  │    Recall    : {m.get('manipulation_recall', 0):.4f}")
                print(f"  │    ROC-AUC   : {m.get('manipulation_roc_auc', 0):.4f}")
                print(f"  │    PR-AUC    : {m.get('manipulation_pr_auc', 0):.4f}")
                tp = m.get('manipulation_tp', 0); fp = m.get('manipulation_fp', 0)
                tn = m.get('manipulation_tn', 0); fn = m.get('manipulation_fn', 0)
                print(f"  │    Confusion : TP={tp}  FP={fp}  TN={tn}  FN={fn}")

            # ── Fake/Real metrics ──────────────────────────────────────────
            if m.get("fake_acc", 0.0) > 0.0 or "fake_roc_auc" in m:
                print(f"  │  [Fake/Real Classification]  n={m.get('n_fake_samples','?')}")
                print(f"  │    Accuracy      : {m.get('fake_acc', 0):.4f}")
                print(f"  │    F1-macro      : {m.get('fake_f1_macro', 0):.4f}")
                print(f"  │    F1-weighted   : {m.get('fake_f1_weighted', 0):.4f}")
                print(f"  │    F1  Real/Fake : {m.get('fake_f1_real', 0):.4f}  /  {m.get('fake_f1_fake', 0):.4f}")
                print(f"  │    Prec Real/Fake: {m.get('fake_precision_real',0):.4f}  /  {m.get('fake_precision_fake',0):.4f}")
                print(f"  │    Rec  Real/Fake: {m.get('fake_recall_real', 0):.4f}  /  {m.get('fake_recall_fake', 0):.4f}")
                print(f"  │    ROC-AUC       : {m.get('fake_roc_auc', 0):.4f}")
                print(f"  │    PR-AUC        : {m.get('fake_pr_auc', 0):.4f}")
                tp = m.get('fake_tp', 0); fp = m.get('fake_fp', 0)
                tn = m.get('fake_tn', 0); fn = m.get('fake_fn', 0)
                print(f"  │    Confusion     : TP={tp}  FP={fp}  TN={tn}  FN={fn}")

            # ── Sentiment metrics ──────────────────────────────────────────
            if "sentiment_acc" in m:
                print(f"  │  [Sentiment Classification]  n={m.get('n_sentiment_samples','?')}")
                print(f"  │    Accuracy      : {m.get('sentiment_acc', 0):.4f}")
                print(f"  │    F1-macro      : {m.get('sentiment_f1_macro', 0):.4f}")
                print(f"  │    F1-weighted   : {m.get('sentiment_f1_weighted', 0):.4f}")
                print(f"  │    Precision-mac : {m.get('sentiment_precision_macro', 0):.4f}")
                print(f"  │    Recall-macro  : {m.get('sentiment_recall_macro', 0):.4f}")
                print(f"  │    Cohen Kappa   : {m.get('sentiment_cohen_kappa', 0):.4f}")
                print(f"  │    Per-class F1  : neg={m.get('sentiment_f1_negative',0):.4f}  "
                      f"neu={m.get('sentiment_f1_neutral',0):.4f}  "
                      f"pos={m.get('sentiment_f1_positive',0):.4f}")
                print(f"  │    Per-class Prec: neg={m.get('sentiment_precision_neg',0):.4f}  "
                      f"neu={m.get('sentiment_precision_neu',0):.4f}  "
                      f"pos={m.get('sentiment_precision_pos',0):.4f}")
                print(f"  │    Per-class Rec : neg={m.get('sentiment_recall_neg',0):.4f}  "
                      f"neu={m.get('sentiment_recall_neu',0):.4f}  "
                      f"pos={m.get('sentiment_recall_pos',0):.4f}")

            print(f"  └{'─'*60}")

        # ── Success criteria check ─────────────────────────────────────────
        print(f"\n  {'─'*60}")
        print(f"  TARGET CHECK")
        print(f"  {'─'*60}")
        # Look for fake/real test — GossipCop or WELFake
        wt = results.get("gossipcop_test", results.get("welfake_test", {}))
        ft_key = "gossipcop_test" if "gossipcop_test" in results else "welfake_test"
        se = results.get("semeval_test", results.get("semeval_dev", {}))
        st = results.get("sentiment_test", results.get("tweet_eval_test", {}))
        st_key = "sentiment_test" if "sentiment_test" in results else "tweet_eval_test"
        ic = results.get("isot_cross_domain", {})
        fa_ok  = wt.get("fake_acc", 0) >= 0.85
        mf_ok  = se.get("manipulation_f1", 0) >= 0.65
        sa_ok  = st.get("sentiment_acc", 0) >= 0.75
        print(f"  FakeAcc   >= 85%  : {wt.get('fake_acc',0):.4f}  {'✅' if fa_ok else '❌'}  [{ft_key}]")
        print(f"  ManipF1   >= 0.65 : {se.get('manipulation_f1',0):.4f}  {'✅' if mf_ok else '❌'}  [semeval test]")
        print(f"  ManipAUC  (ref)   : {se.get('manipulation_roc_auc',0):.4f}")
        print(f"  SentAcc   >= 75%  : {st.get('sentiment_acc',0):.4f}  {'✅' if sa_ok else '❌'}  [{st_key}]")
        # LIAR cross-domain (political statements — tests generalization beyond GossipCop)
        li = results.get("liar_test", {})
        if li:
            li_ok = li.get("fake_acc", 0) >= 0.55
            print(f"  LIAR cross-domain : {li.get('fake_acc',0):.4f}  {'✅' if li_ok else '⚠️ '} "
                  f"ROC-AUC={li.get('fake_roc_auc',0):.4f}  [GossipCop+LIAR→LIAR]")
        if ic:
            xd_ok = ic.get("fake_acc", 0) >= 0.55
            print(f"  ISOT cross-domain : {ic.get('fake_acc',0):.4f}  {'✅' if xd_ok else '⚠️ '} "
                  f"ROC-AUC={ic.get('fake_roc_auc',0):.4f}  [trained on GossipCop+LIAR+ISOT_train → tested on ISOT_test]")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint save / load
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, f1: float) -> None:
        out_dir = Path(self.cfg.output_dir)
        torch.save({"epoch": epoch, "f1": f1, "model_state": self.model.state_dict(), "config": self.cfg}, out_dir / "best_model.pt")
        self.tokenizer.save_pretrained(out_dir / "tokenizer")
        if self._style_scaler is not None and self._style_scaler.mean_ is not None:
            self._style_scaler.save(str(out_dir / "style_scaler"))

    def load_best_model(self) -> None:
        ckpt_path = Path(self.cfg.output_dir) / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping load")
            return
        if self.model is None:
            self.build_model()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        scaler_path = ckpt_path.parent / "style_scaler.npz"
        if scaler_path.exists() and self.cfg.normalize_style:
            self._style_scaler = StyleScaler.load(str(scaler_path))

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_style(self, text: str) -> torch.Tensor:
        raw = StylometricExtractor.extract(text)
        if self._style_scaler is not None and self._style_scaler.mean_ is not None:
            raw = self._style_scaler.transform(raw.reshape(1, -1)).squeeze(0)
        return torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, text: str) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        enc = self.tokenizer(text, max_length=self.cfg.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")
        style = self._extract_style(text)
        with torch.no_grad(), autocast("cuda", enabled=self.cfg.fp16):
            out = self.model.get_predictions(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), style)
        fake_score = out["fake_prob"].item()
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        manip_score = out["manipulation_score"].item()
        return {
            "text": text,
            "fake_score": fake_score,
            "fake_class": "fake" if out["fake_class"].item() == 1 else "real",
            "sentiment_class": sentiment_map[out["sentiment_class"].item()],
            "sentiment_intensity": out["sentiment_intensity"].item(),
            "manipulation_score": manip_score,
            "manipulation_class": "manipulative" if manip_score >= self._manip_threshold else "normal",
            "manipulation_threshold": self._manip_threshold,
            "manipulation_vector": out["manipulation_vector"].squeeze(0).cpu(),
            "summary": f"This news is {manip_score*100:.1f}% manipulative (threshold={self._manip_threshold:.2f}), {fake_score*100:.1f}% likely fake.",
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        all_fake, all_sent_class, all_sent_int, all_manip, all_vec = [], [], [], [], []
        for i in range(0, len(texts), self.cfg.gnn_batch_size):
            batch_texts = texts[i : i + self.cfg.gnn_batch_size]
            enc = self.tokenizer(batch_texts, max_length=self.cfg.max_seq_len, padding=True, truncation=True, return_tensors="pt")
            raw_style = StylometricExtractor.batch_extract(batch_texts)
            if self._style_scaler is not None and self._style_scaler.mean_ is not None:
                raw_style = self._style_scaler.transform(raw_style)
            style = torch.tensor(raw_style, dtype=torch.float32).to(self.device)
            with autocast("cuda", enabled=self.cfg.fp16):
                out = self.model.get_predictions(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), style)
            all_fake.append(out["fake_prob"].cpu())
            all_sent_class.append(out["sentiment_class"].cpu())
            all_sent_int.append(out["sentiment_intensity"].cpu())
            all_manip.append(out["manipulation_score"].cpu())
            all_vec.append(out["manipulation_vector"].cpu())
        return {
            "fake_scores": torch.cat(all_fake, dim=0),
            "sentiment_classes": torch.cat(all_sent_class, dim=0),
            "sentiment_intensities": torch.cat(all_sent_int, dim=0),
            "manipulation_scores": torch.cat(all_manip, dim=0),
            "manipulation_vectors": torch.cat(all_vec, dim=0),
        }


def run_full_pipeline(cfg: Optional[TrainerConfig] = None) -> Model1ExpertTrainer:
    from src.features.gnn_exporter import GNNFeatureExporter

    trainer = Model1ExpertTrainer(cfg)
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.load_best_model()
    try:
        GNNFeatureExporter(trainer).export()
    except FileNotFoundError:
        pass
    return trainer
