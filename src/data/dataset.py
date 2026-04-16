"""
PyTorch Dataset and DataModule for the HuffPost News Category dataset.
"""
import json
import logging
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """Wraps encoded sequences and integer labels as a PyTorch Dataset."""

    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class SentimentDataModule:
    """
    Handles end-to-end data preparation:
      1. Load the raw JSONL file.
      2. Map categories to 3-class sentiment labels.
      3. Build vocabulary from the training split.
      4. Encode all headlines.
      5. Return train / val / test NewsDataset objects.
    """

    def __init__(self, config: dict, preprocessor: TextPreprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def load_and_prepare(
        self, path: str
    ) -> Tuple[NewsDataset, NewsDataset, NewsDataset]:
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

        # ── Stratified split: 80 % train / 10 % val / 10 % test ──────────
        test_size = self.config["data"]["test_size"]      # e.g. 0.20
        val_size  = self.config["data"]["val_size"]       # e.g. 0.10

        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size,
            random_state=42, stratify=df["label"]
        )
        val_frac = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1 - val_frac,
            random_state=42, stratify=temp_df["label"]
        )

        # ── Build vocabulary only from training headlines ─────────────────
        self.preprocessor.build_vocab(train_df["headline"].tolist())

        # ── Encode all splits ─────────────────────────────────────────────
        for split_df in (train_df, val_df, test_df):
            split_df["sequence"] = split_df["headline"].apply(
                self.preprocessor.encode
            )

        logger.info(
            f"Splits → train: {len(train_df):,} | "
            f"val: {len(val_df):,} | test: {len(test_df):,}"
        )

        return (
            NewsDataset(train_df["sequence"].tolist(), train_df["label"].tolist()),
            NewsDataset(val_df["sequence"].tolist(),   val_df["label"].tolist()),
            NewsDataset(test_df["sequence"].tolist(),  test_df["label"].tolist()),
        )
