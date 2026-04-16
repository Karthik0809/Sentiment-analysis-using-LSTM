"""Unit tests for model architectures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from src.models.lstm import BiLSTMWithAttention, BaselineLSTM, StackedBiGRU

VOCAB_SIZE   = 1000
EMBED_DIM    = 32
HIDDEN_DIM   = 64
NUM_CLASSES  = 3
N_LAYERS     = 2
BATCH_SIZE   = 4
SEQ_LEN      = 20


@pytest.fixture
def dummy_input():
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))


def _common_kwargs():
    return dict(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        n_layers=N_LAYERS,
        dropout=0.0,
    )


class TestBiLSTMWithAttention:
    def test_output_shape(self, dummy_input):
        model = BiLSTMWithAttention(**_common_kwargs())
        logits, attn = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert attn.shape == (BATCH_SIZE, SEQ_LEN)

    def test_attention_sums_to_one(self, dummy_input):
        model = BiLSTMWithAttention(**_common_kwargs())
        _, attn = model(dummy_input)
        sums = attn.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_no_nan_in_logits(self, dummy_input):
        model = BiLSTMWithAttention(**_common_kwargs())
        logits, _ = model(dummy_input)
        assert not torch.isnan(logits).any()


class TestBaselineLSTM:
    def test_output_shape(self, dummy_input):
        model = BaselineLSTM(**_common_kwargs())
        logits, attn = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert attn is None

    def test_no_nan(self, dummy_input):
        model = BaselineLSTM(**_common_kwargs())
        logits, _ = model(dummy_input)
        assert not torch.isnan(logits).any()


class TestStackedBiGRU:
    def test_output_shape(self, dummy_input):
        model = StackedBiGRU(**_common_kwargs())
        logits, attn = model(dummy_input)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert attn is None
