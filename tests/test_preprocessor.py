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
    assert pp.get_label("Team wins championship in stunning victory") == 2
    assert pp.get_label("Amazing breakthrough brings hope to millions") == 2


def test_label_negative():
    pp = TextPreprocessor()
    assert pp.get_label("Stock market crashes amid recession fears") == 0
    assert pp.get_label("Deadly attack kills dozens in tragic shooting") == 0


def test_label_neutral():
    pp = TextPreprocessor()
    assert pp.get_label("Scientists publish new study on climate data") == 1
    assert pp.get_label("Government announces budget review committee") == 1


def test_label_returns_valid_class():
    pp = TextPreprocessor()
    for headline in [
        "Markets rise on positive jobs report",
        "Flood warnings issued across three states",
        "New policy takes effect next month",
    ]:
        assert pp.get_label(headline) in (0, 1, 2)


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
