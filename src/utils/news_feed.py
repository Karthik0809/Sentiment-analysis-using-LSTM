"""
Live news headline fetcher using public RSS feeds.
Requires: pip install feedparser

Supported sources:
  bbc        → BBC News Top Stories
  npr        → NPR News
  cnn        → CNN Top Stories
  techcrunch → TechCrunch
  guardian   → The Guardian World
  aljazeera  → Al Jazeera English
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Public RSS feed URLs for major news outlets
RSS_FEEDS: Dict[str, str] = {
    "BBC News":      "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR":           "https://feeds.npr.org/1001/rss.xml",
    "CNN":           "http://rss.cnn.com/rss/cnn_topstories.rss",
    "TechCrunch":    "https://techcrunch.com/feed/",
    "The Guardian":  "https://www.theguardian.com/world/rss",
    "Al Jazeera":    "https://www.aljazeera.com/xml/rss/all.xml",
}


@dataclass
class NewsHeadline:
    title: str
    source: str
    published: str = ""
    link: str = ""
    summary: str = ""
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)


def fetch_headlines(
    source_name: str,
    max_items: int = 30,
    timeout: int = 10,
) -> List[NewsHeadline]:
    """
    Fetch headlines from a named RSS source.

    Parameters
    ----------
    source_name : key from RSS_FEEDS dict (e.g. "BBC News")
    max_items   : maximum headlines to return
    timeout     : request timeout in seconds

    Returns
    -------
    List of NewsHeadline objects (title + metadata; sentiment not yet set)
    """
    try:
        import feedparser
    except ImportError:
        raise ImportError("feedparser is required: pip install feedparser")

    url = RSS_FEEDS.get(source_name)
    if not url:
        raise ValueError(
            f"Unknown source '{source_name}'. "
            f"Available: {list(RSS_FEEDS.keys())}"
        )

    logger.info(f"Fetching RSS: {source_name} → {url}")
    feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})

    if feed.bozo:
        logger.warning(f"RSS parse warning for {source_name}: {feed.bozo_exception}")

    headlines: List[NewsHeadline] = []
    for entry in feed.entries[:max_items]:
        title = getattr(entry, "title", "").strip()
        if not title:
            continue
        published = ""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published = datetime(*entry.published_parsed[:6]).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                published = getattr(entry, "published", "")
        headlines.append(
            NewsHeadline(
                title=title,
                source=source_name,
                published=published,
                link=getattr(entry, "link", ""),
                summary=getattr(entry, "summary", "")[:200],
            )
        )

    logger.info(f"Fetched {len(headlines)} headlines from {source_name}")
    return headlines


def fetch_multi_source(
    sources: List[str],
    max_per_source: int = 20,
) -> List[NewsHeadline]:
    """Fetch from multiple sources; silently skip sources that fail."""
    all_headlines: List[NewsHeadline] = []
    for source in sources:
        try:
            all_headlines.extend(fetch_headlines(source, max_items=max_per_source))
        except Exception as exc:
            logger.warning(f"Skipping {source}: {exc}")
    return all_headlines


def analyze_headlines(
    headlines: List[NewsHeadline],
    model,
    preprocessor,
    device,
    tokenizer=None,
) -> List[NewsHeadline]:
    """
    Run batch sentiment inference on a list of NewsHeadline objects.
    Populates .sentiment, .confidence, and .scores in-place.

    Parameters
    ----------
    headlines    : list from fetch_headlines / fetch_multi_source
    model        : trained PyTorch model (returns logits, attn_or_None)
    preprocessor : fitted TextPreprocessor (used when tokenizer is None)
    device       : torch.device
    tokenizer    : HuggingFace tokenizer — pass for DistilBERT models,
                   omit (or pass None) for BiLSTM models
    """
    import torch

    LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

    if not headlines:
        return headlines

    titles = [h.title for h in headlines]

    with torch.no_grad():
        if tokenizer is not None:
            # DistilBERT path
            enc = tokenizer(
                titles,
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            logits, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        else:
            # BiLSTM path
            sequences = [preprocessor.encode(t) for t in titles]
            tensor = torch.tensor(sequences, dtype=torch.long).to(device)
            logits, _ = model(tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()

    for i, headline in enumerate(headlines):
        pred = int(probs[i].argmax())
        headline.sentiment   = LABEL_MAP[pred]
        headline.confidence  = float(probs[i][pred])
        headline.scores      = {LABEL_MAP[j]: float(probs[i][j]) for j in range(3)}

    return headlines
