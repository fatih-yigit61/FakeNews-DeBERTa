import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

from configs.config import MAX_SEQ_LEN
from src.features.stylometry import StyleScaler, StylometricExtractor


@dataclass
class SentenceSample:
    article_id: str
    sentence_id: int
    text: str
    label: int
    char_start: int
    char_end: int


class SemEvalParser:
    LABEL_PATTERN = re.compile(r"^(\d+)\t(.+?)\t(\d+)\t(\d+)$")

    def __init__(self, articles_dir: str, labels_dir: str) -> None:
        self.articles_dir = Path(articles_dir)
        self.labels_dir = Path(labels_dir)

    def _parse_labels(self, article_id: str) -> List[Tuple[int, int]]:
        label_files = list(self.labels_dir.glob(f"article{article_id}.*"))
        if not label_files:
            return []
        spans = []
        with open(label_files[0], encoding="utf-8") as f:
            for line in f:
                m = self.LABEL_PATTERN.match(line.strip())
                if m:
                    spans.append((int(m.group(3)), int(m.group(4))))
        return spans

    @staticmethod
    def _overlaps(s_start: int, s_end: int, spans: List[Tuple[int, int]]) -> bool:
        return any(max(s_start, e_start) < min(s_end, e_end) for e_start, e_end in spans)

    def parse(self) -> List[SentenceSample]:
        article_files = sorted(self.articles_dir.glob("article*.txt"))
        samples: List[SentenceSample] = []
        for art_file in tqdm(article_files, desc="Parsing articles"):
            article_id = art_file.stem.replace("article", "")
            text = art_file.read_text(encoding="utf-8")
            spans = self._parse_labels(article_id)
            sents = nltk.sent_tokenize(text)
            offset = 0
            for i, s in enumerate(sents):
                idx = text.find(s, offset)
                if idx == -1:
                    # Whitespace-normalised retry before silent fallback
                    s_norm = " ".join(s.split())
                    idx = text.find(s_norm, offset)
                if idx == -1:
                    log.debug("Sentence boundary not found in article %s at offset %d: %.60s", article_id, offset, s)
                    idx = offset
                s_start, s_end = idx, idx + len(s)
                offset = s_end
                if len(s.strip()) < 5:
                    continue
                label = 1 if (spans and self._overlaps(s_start, s_end, spans)) else 0
                samples.append(SentenceSample(article_id, i, s.strip(), label, s_start, s_end))
        return samples


class PropagandaDataset(Dataset):
    """SemEval propaganda sentence dataset.

    SemEval provides only span-level manipulation annotations — it has no fake-news
    or sentiment labels.  ``fake_label`` and ``sentiment_label`` are therefore set
    to **-1** (sentinel), which causes ``MultiTaskLoss`` to mask those loss terms
    for these samples.  Only ``manipulation_label`` contributes to training here.
    """

    def __init__(
        self,
        samples: List[SentenceSample],
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
        style_scaler: Optional[StyleScaler] = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.style_scaler = style_scaler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        enc = self.tokenizer(sample.text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        raw_style = StylometricExtractor.extract(sample.text)
        if self.style_scaler is not None:
            raw_style = self.style_scaler.transform(raw_style.reshape(1, -1)).squeeze(0)
        style = torch.tensor(raw_style, dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "style_feats": style,
            "manipulation_label": torch.tensor(sample.label, dtype=torch.float32),
            "fake_label": torch.tensor(-1, dtype=torch.long),          # no fake label in SemEval
            "sentiment_label": torch.tensor(-1, dtype=torch.long),     # no sentiment label in SemEval
            "sentiment_intensity": torch.tensor(0.0, dtype=torch.float32),
        }


class SentimentDataset(Dataset):
    """Sentiment dataset for training the sentiment_class_head and sentiment_intensity_head.

    Expects 3-class labels: 0=negative, 1=neutral, 2=positive (tweet_eval convention).
    ``sentiment_intensity`` is derived from the label:
        negative → 0.1,  neutral → 0.5,  positive → 0.9
    This avoids needing a separate continuous-score annotation while still providing
    a meaningful gradient signal for the intensity head.

    ``fake_label`` and ``manipulation_label`` are set to -1 (sentinel) so
    MultiTaskLoss masks those loss terms for these samples.
    """

    _INTENSITY = {0: 0.1, 1: 0.5, 2: 0.9}

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tok,
        max_len: int = MAX_SEQ_LEN,
        style_scaler: Optional[StyleScaler] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len
        self.style_scaler = style_scaler

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tok(self.texts[idx], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        raw_style = StylometricExtractor.extract(self.texts[idx])
        if self.style_scaler is not None:
            raw_style = self.style_scaler.transform(raw_style.reshape(1, -1)).squeeze(0)
        label = self.labels[idx]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "style_feats": torch.tensor(raw_style, dtype=torch.float32),
            "fake_label": torch.tensor(-1, dtype=torch.long),
            "manipulation_label": torch.tensor(-1, dtype=torch.float32),
            "sentiment_label": torch.tensor(label, dtype=torch.long),
            "sentiment_intensity": torch.tensor(self._INTENSITY[label], dtype=torch.float32),
        }


class SimpleNewsDataset(Dataset):
    """Generic fake-news dataset (WELFake, ISOT, etc.).

    Schema is kept identical to PropagandaDataset so both datasets can be used
    with a ConcatDataset in multi-source training without a custom collate_fn.

    ``fake_label`` carries the real 0/1 annotation from the source dataset.
    ``manipulation_label`` and ``sentiment_label`` are set to sentinel -1 because
    these datasets do not provide that annotation — MultiTaskLoss will mask them.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tok,
        max_len: int = MAX_SEQ_LEN,
        style_scaler: Optional[StyleScaler] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len
        self.style_scaler = style_scaler
        log.info("SimpleNewsDataset: %d samples", len(texts))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tok(self.texts[idx], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        raw_style = StylometricExtractor.extract(self.texts[idx])
        if self.style_scaler is not None:
            raw_style = self.style_scaler.transform(raw_style.reshape(1, -1)).squeeze(0)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "style_feats": torch.tensor(raw_style, dtype=torch.float32),
            "fake_label": torch.tensor(self.labels[idx], dtype=torch.long),
            "manipulation_label": torch.tensor(-1, dtype=torch.float32),  # no manip label
            "sentiment_label": torch.tensor(-1, dtype=torch.long),        # no sentiment label
            "sentiment_intensity": torch.tensor(0.0, dtype=torch.float32),
        }
