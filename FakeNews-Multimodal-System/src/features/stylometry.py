import logging
import re
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"[.!?]+")


class StylometricExtractor:
    _QUOTE_RE = re.compile(
        r'["\']([^"\']{4,})["\']'
        r"|\u201c([^\u201d]+)\u201d"
        r"|\u2018([^\u2019]+)\u2019"
        r"|\u00ab([^\u00bb]+)\u00bb",
        re.UNICODE,
    )
    # Normalisation constant for avg sentence length (typical news sentence ≈ 20 words)
    _AVG_SENT_LEN_NORM = 20.0

    @classmethod
    def extract(cls, text: str) -> np.ndarray:
        words = text.split()
        n_words = max(1, len(words))
        n_chars = max(1, sum(c.isalpha() for c in text))

        # Features XLM-R can capture implicitly (surface signals)
        punct_density = (text.count("!") + text.count("?")) / n_words
        caps_ratio = sum(1 for c in text if c.isupper()) / n_chars
        quotation_flag = 1.0 if cls._QUOTE_RE.search(text) else 0.0

        # Higher-level features XLM-R cannot directly model (discourse-level)
        # TTR (type-token ratio): lexical diversity — fake news often repeats vocabulary.
        # NOTE: TTR approaches 1.0 for very short texts (< ~15 words); it is most
        # discriminative at document level (WELFake/ISOT full articles).  StyleScaler
        # z-score normalisation reduces its variance impact at sentence level (SemEval).
        ttr = len({w.lower() for w in words}) / n_words
        # Avg sentence length normalised by a typical news-sentence length (20 words)
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        avg_sent_len_norm = (n_words / max(1, len(sentences))) / cls._AVG_SENT_LEN_NORM

        return np.array([punct_density, caps_ratio, quotation_flag, ttr, avg_sent_len_norm], dtype=np.float32)

    @classmethod
    def batch_extract(cls, texts: List[str]) -> np.ndarray:
        return np.stack([cls.extract(t) for t in texts])


class StyleScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StyleScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        log.info(f"StyleScaler fit: mean={self.mean_.round(4)} std={self.std_.round(4)}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "StyleScaler.fit() çağrılmamış!"
        return (X - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> "StyleScaler":
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.std_ = data["std"]
        return scaler
