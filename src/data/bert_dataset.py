"""
PyTorch Dataset and DataModule for DistilBERT fine-tuning.

Uses the same category-based label mapping as SentimentDataModule but
tokenises headlines with the DistilBERT fast tokenizer instead of building
a custom vocabulary.
"""
import json
import logging
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class BertNewsDataset(Dataset):
    """HuggingFace-tokenised headlines as a PyTorch Dataset."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 64):
        self.labels = torch.tensor(labels, dtype=torch.long)
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          self.labels[idx],
        }


class BertDataModule:
    """
    Same stratified 80/10/10 split logic as SentimentDataModule, but
    produces BertNewsDataset objects.
    """

    def __init__(self, config: dict, tokenizer, preprocessor: TextPreprocessor):
        self.config       = config
        self.tokenizer    = tokenizer
        self.preprocessor = preprocessor

    def load_and_prepare(
        self, path: str, max_len: int = 64
    ) -> Tuple[BertNewsDataset, BertNewsDataset, BertNewsDataset]:
        logger.info(f"Loading dataset from {path}")
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))

        df = pd.DataFrame(records)
        df["label"] = df["category"].apply(self.preprocessor.get_label)
        df = df[df["label"] != -1].reset_index(drop=True)

        logger.info(
            f"Dataset size: {len(df):,} | Label distribution:\n"
            + str(df["label"].value_counts().to_dict())
        )

        test_size = self.config["data"]["test_size"]
        val_size  = self.config["data"]["val_size"]

        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size,
            random_state=42, stratify=df["label"],
        )
        val_frac = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1 - val_frac,
            random_state=42, stratify=temp_df["label"],
        )

        logger.info(
            f"Splits → train: {len(train_df):,} | "
            f"val: {len(val_df):,} | test: {len(test_df):,}"
        )

        return (
            BertNewsDataset(train_df["headline"].tolist(), train_df["label"].tolist(), self.tokenizer, max_len),
            BertNewsDataset(val_df["headline"].tolist(),   val_df["label"].tolist(),   self.tokenizer, max_len),
            BertNewsDataset(test_df["headline"].tolist(),  test_df["label"].tolist(),  self.tokenizer, max_len),
        )
