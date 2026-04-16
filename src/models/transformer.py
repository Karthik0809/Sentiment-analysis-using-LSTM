"""
DistilBERT-based sentiment classifier for comparison with recurrent models.
Uses HuggingFace Transformers — requires: pip install transformers
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DistilBERTSentiment(nn.Module):
    """
    DistilBERT fine-tuned for 3-class sentiment classification.

    Compared to the BiLSTM+Attention model this is ~40 % smaller than
    BERT-base and runs ~60 % faster at inference, while retaining 97 %
    of BERT's performance (Sanh et al., 2019).

    The [CLS] token representation is passed through a two-layer MLP head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        try:
            from transformers import DistilBertModel
        except ImportError:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers"
            )

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.classifier(self.dropout(cls_repr))
        return logits, None


class BERTNewsDataset(torch.utils.data.Dataset):
    """Dataset wrapper that tokenizes headlines using a HuggingFace tokenizer."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 64):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }
