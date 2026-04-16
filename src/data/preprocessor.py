"""
Text preprocessing pipeline for news headline sentiment analysis.
Handles tokenization, lemmatization, stopword removal, and vocabulary building.
"""
import re
import logging
from typing import List, Dict
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _resource in ["punkt", "stopwords", "wordnet", "punkt_tab", "omw-1.4"]:
    nltk.download(_resource, quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Full preprocessing pipeline: clean → tokenize → lemmatize → encode.

    Supports three-class (Negative=0, Neutral=1, Positive=2) labeling
    based on HuffPost news categories.
    """

    POSITIVE_CATEGORIES = {
        "WELLNESS", "ENTERTAINMENT", "FOOD & DRINK", "HEALTHY LIVING",
        "SPORTS", "GOOD NEWS", "COMEDY", "TRAVEL", "ARTS & CULTURE",
        "HOME & LIVING", "STYLE & BEAUTY", "PARENTING", "WEDDINGS",
        "QUEER VOICES", "WOMEN",
    }
    NEGATIVE_CATEGORIES = {
        "POLITICS", "CRIME", "DIVORCE", "WORLD NEWS", "U.S. NEWS",
        "MEDIA", "WEIRD NEWS", "IMPACT", "BLACK VOICES",
    }
    NEUTRAL_CATEGORIES = {
        "SCIENCE", "TECH", "BUSINESS", "MONEY", "EDUCATION",
        "RELIGION", "COLLEGE", "ARTS", "ENVIRONMENT", "FIFTY",
        "LATINO VOICES", "THE WORLDPOST", "WORLDPOST", "CULTURE & ARTS",
        "TASTE", "PARENTS",
    }

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_len: int = 50, vocab_size: int = 50_000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.word_counts: Counter = Counter()
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------------

    def get_label(self, category: str) -> int:
        """Map a HuffPost category string to a sentiment integer (0/1/2)."""
        if category in self.POSITIVE_CATEGORIES:
            return 2
        if category in self.NEGATIVE_CATEGORIES:
            return 0
        if category in self.NEUTRAL_CATEGORIES:
            return 1
        return -1  # unknown / excluded

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [
            self._lemmatizer.lemmatize(t)
            for t in tokens
            if t.isalpha() and t not in self._stop_words and len(t) > 1
        ]

    def preprocess(self, text: str) -> List[str]:
        """Return a list of preprocessed tokens for one headline."""
        return self.tokenize(self.clean(text))

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocab(self, texts: List[str]) -> None:
        """Build word→index vocabulary from a list of raw text strings."""
        all_tokens: List[str] = []
        for t in texts:
            all_tokens.extend(self.preprocess(t))
        self.word_counts = Counter(all_tokens)

        # index 0 = PAD, 1 = UNK, then most-frequent words
        top_words = self.word_counts.most_common(self.vocab_size - 2)
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            **{word: idx + 2 for idx, (word, _) in enumerate(top_words)},
        }
        logger.info(f"Vocabulary built: {len(self.vocab):,} tokens")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert a raw text string to a zero-padded integer sequence."""
        tokens = self.preprocess(text)
        ids = [self.vocab.get(t, 1) for t in tokens]  # 1 = UNK
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))         # zero-pad
        return ids

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab) + 1
