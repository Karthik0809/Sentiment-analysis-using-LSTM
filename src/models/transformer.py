"""
DistilBERT fine-tuning wrapper for 3-class news headline sentiment.

Exposes the same (logits, None) return signature as BiLSTMWithAttention so
it can be used as a drop-in replacement throughout the codebase.
"""
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class DistilBertSentiment(nn.Module):
    """
    Thin wrapper around HuggingFace DistilBertForSequenceClassification.

    Returns (logits, None) to match the BiLSTM interface — callers that
    unpack attention weights will simply receive None for this model.
    """

    def __init__(
        self,
        num_labels: int = 3,
        pretrained: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, None   # None keeps interface compatible

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        """Save model weights and config via HuggingFace save_pretrained."""
        self.bert.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str) -> "DistilBertSentiment":
        """Load a previously fine-tuned model from a local directory or Hub ID."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.bert = DistilBertForSequenceClassification.from_pretrained(path)
        return instance
