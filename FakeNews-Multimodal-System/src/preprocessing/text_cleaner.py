import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CleanerConfig:
    preserve_casing: bool = True
    min_text_length: int = 5


class AdvancedTextCleaner:
    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self.cfg = config or CleanerConfig()

    def detect_language(self, text: str) -> tuple[str, float]:
        return ("tr", 0.7) if text and len(text.strip()) >= self.cfg.min_text_length else ("tr", 0.0)

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        mentions = re.findall(r"@[A-Za-z0-9_]+", text)
        urls = re.findall(r"https?://\\S+|www\\.\\S+", text)
        hashtags = re.findall(r"#(\\w+)", text)
        return {"mentions": mentions, "urls": urls, "hashtags": hashtags}

    def clean_text(self, text: str, lang: Optional[str] = None) -> str:
        out = re.sub(r"https?://\\S+|www\\.\\S+", "[LINK]", text)
        out = re.sub(r"@[A-Za-z0-9_]+", "[USER]", out)
        out = re.sub(r"#(\\w+)", r"\\1", out)
        if not self.cfg.preserve_casing:
            out = out.lower()
        return out.strip()


# ── Source-identifying markers that leak dataset origin ──────────────────────
# The model exploits these to distinguish gossip-site style vs mainstream media
# style instead of learning actual fake-news semantics.  Masking forces the model
# to rely on content/claim plausibility rather than surface source cues.

_NEWS_SOURCES = sorted([
    # Mainstream / real-leaning
    "Reuters", "Associated Press", "AP News", "CNN", "BBC", "The New York Times",
    "New York Times", "NYT", "The Washington Post", "Washington Post", "NPR",
    "The Guardian", "Al Jazeera", "Bloomberg", "CNBC", "ABC News", "CBS News",
    "NBC News", "Fox News", "USA Today", "The Wall Street Journal",
    "Wall Street Journal", "WSJ", "Los Angeles Times", "Chicago Tribune",
    "The Independent", "The Telegraph", "Sky News", "HuffPost", "Huffington Post",
    "Politico", "The Hill", "Axios", "Vox", "BuzzFeed News", "BuzzFeed",
    "The Daily Beast", "Vice News", "VICE",
    # Gossip / tabloid / fake-leaning sources in GossipCop
    "GossipCop", "Gossip Cop", "TMZ", "Page Six", "PageSix", "E! News",
    "Entertainment Tonight", "People Magazine", "People", "Us Weekly",
    "Hollywood Life", "HollywoodLife", "RadarOnline", "Radar Online",
    "Star Magazine", "OK! Magazine", "In Touch Weekly", "Life & Style",
    "National Enquirer", "The Inquisitr", "Inquisitr", "CelebrityDirtyLaundry",
    "Celebrity Dirty Laundry", "MediaMass", "Empire News", "Huzlers",
    "World News Daily Report", "The Hollywood Gossip", "Perez Hilton",
    "Naughty Gossip", "Celeb Dirty Laundry", "Gossip Extra", "GossipExtra",
    "Just Jared", "ET Online", "E! Online", "Access Hollywood",
    # Political / fact-check sources in ISOT/LIAR
    "PolitiFact", "Snopes", "FactCheck.org", "Breitbart", "InfoWars",
    "The Daily Caller", "Daily Mail", "Daily Express", "The Sun",
    "New York Post", "RT", "Russia Today", "Sputnik", "TASS",
], key=len, reverse=True)  # longest first to avoid partial matches

# Pre-compile a single regex for speed (case-insensitive)
_SOURCE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _NEWS_SOURCES) + r")\b",
    re.IGNORECASE,
)

# Byline patterns: "By John Smith", "reported by Jane Doe for CNN"
_BYLINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:By|Written by|Reported by|Author:)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r"(?:\s*(?:,|for|/|–|-)\s*\S+)?",
    re.IGNORECASE,
)

# Dateline: "NEW YORK (Reuters) —", "WASHINGTON, March 15 (AP) —"
_DATELINE_PATTERN = re.compile(
    r"^[A-Z\s,]+(?: \([^)]+\))?\s*[-—–]\s*",
    re.MULTILINE,
)

# URL pattern
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def debias_source(text: str) -> str:
    """Remove source-identifying markers so the model cannot shortcut on source style.

    Masks:
    - News organization names → [SOURCE]
    - Bylines (By Author Name) → empty
    - Datelines (CITY (Agency) —) → empty
    - URLs → [LINK]

    This is the KEY anti-shortcut measure for the fake/real head.
    Without this, the model learns "Reuters style = real, GossipExtra style = fake"
    instead of learning content-level fake news signals.
    """
    if not text:
        return text
    out = _URL_PATTERN.sub("[LINK]", text)
    out = _DATELINE_PATTERN.sub("", out)
    out = _BYLINE_PATTERN.sub("", out)
    out = _SOURCE_PATTERN.sub("[SOURCE]", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def normalize_style(text: str) -> str:
    """Reduce stylistic shortcuts in fake-news datasets.

    Targets surface-level cues that let models distinguish source style
    rather than content veracity:
    - Repeated punctuation: '!!!' → '!', '???' → '?'
    - ALL CAPS words (4+ letters): 'BREAKING NEWS' → 'Breaking News'
    - Excessive whitespace collapsed to single space
    """
    if not text:
        return text
    # Collapse repeated punctuation (keep one)
    out = re.sub(r"([!?.])\1{1,}", r"\1", text)
    # Convert ALL CAPS words (4+ letters) to title case
    # Preserves short acronyms like "US", "EU", "FBI"
    out = re.sub(
        r"\b([A-Z]{4,})\b",
        lambda m: m.group(1).capitalize(),
        out,
    )
    # Collapse whitespace
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def clean_for_fake_detection(text: str) -> str:
    """Full cleaning pipeline for fake/real classification.

    Combines source debiasing + style normalization to remove
    ALL known shortcuts.  Applied to GossipCop, LIAR, ISOT, WELFake.
    """
    text = debias_source(text)
    text = normalize_style(text)
    return text
