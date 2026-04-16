"""Unit tests for the TextPreprocessor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.data.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    pp = TextPreprocessor(max_len=10, vocab_size=100)
    texts = [
        "Scientists discover cure for disease",
        "Stock market crashes amid fears",
        "Local team wins championship award",
    ]
    pp.build_vocab(texts)
    return pp


def test_clean_removes_urls():
    pp = TextPreprocessor()
    cleaned = pp.clean("Check https://example.com for more info!!!")
    assert "https" not in cleaned
    assert "!!!" not in cleaned


def test_label_positive():
    pp = TextPreprocessor()
    assert pp.get_label("WELLNESS") == 2
    assert pp.get_label("SPORTS") == 2


def test_label_negative():
    pp = TextPreprocessor()
    assert pp.get_label("POLITICS") == 0
    assert pp.get_label("CRIME") == 0


def test_label_neutral():
    pp = TextPreprocessor()
    assert pp.get_label("SCIENCE") == 1
    assert pp.get_label("TECH") == 1


def test_label_unknown():
    pp = TextPreprocessor()
    assert pp.get_label("UNKNOWN_CATEGORY") == -1


def test_encode_length(preprocessor):
    seq = preprocessor.encode("Scientists discover amazing breakthrough")
    assert len(seq) == preprocessor.max_len


def test_encode_padding(preprocessor):
    seq = preprocessor.encode("Win")
    # Short headline should be zero-padded
    assert seq[-1] == 0


def test_vocab_has_pad_unk(preprocessor):
    assert "<PAD>" in preprocessor.vocab
    assert "<UNK>" in preprocessor.vocab
    assert preprocessor.vocab["<PAD>"] == 0
    assert preprocessor.vocab["<UNK>"] == 1
