"""
Pre-trained word embedding utilities.

Supports GloVe (Stanford NLP) and FastText (Facebook AI) vectors.

Quick start
-----------
# 1. Download GloVe 100-dim vectors (822 MB zip, 171 MB extracted)
python scripts/download_glove.py

# 2. Train with pre-trained embeddings
python scripts/train.py --glove-path data/glove/glove.6B.100d.txt

Coverage typically reaches 85–92 % of the training vocabulary.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_glove(
    glove_path: str,
    vocab: Dict[str, int],
    embedding_dim: int,
    freeze: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Build an embedding matrix from GloVe vectors.

    Words in *vocab* that appear in the GloVe file get their pre-trained
    vector; all others keep a small random initialisation (mean=0, std=0.1).
    Index 0 (PAD) is always kept as the zero vector.

    Parameters
    ----------
    glove_path    : path to the GloVe .txt file (e.g. glove.6B.100d.txt)
    vocab         : word → integer index mapping from TextPreprocessor
    embedding_dim : must match the dim in the GloVe file
    freeze        : if True the embedding layer will not be updated during training

    Returns
    -------
    embedding_matrix : FloatTensor of shape (vocab_size+1, embedding_dim)
    n_found          : number of vocabulary words matched in GloVe
    """
    glove_path = Path(glove_path)
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {glove_path}\n"
            "Run: python scripts/download_glove.py"
        )

    vocab_size = len(vocab) + 1            # +1 for the zero row at index 0
    matrix = np.random.normal(0.0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)
    matrix[0] = 0.0                        # PAD → zero vector

    logger.info(f"Loading GloVe vectors from {glove_path} …")
    n_found = 0
    with open(glove_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                matrix[idx] = np.array(parts[1:], dtype=np.float32)
                n_found += 1

    coverage = n_found / max(len(vocab), 1) * 100
    logger.info(
        f"GloVe coverage: {n_found:,}/{len(vocab):,} vocabulary tokens "
        f"({coverage:.1f} %)"
    )

    return torch.tensor(matrix, dtype=torch.float32), n_found


def load_fasttext(
    fasttext_path: str,
    vocab: Dict[str, int],
    embedding_dim: int,
) -> Tuple[torch.Tensor, int]:
    """
    Same interface as load_glove but for FastText .vec files.
    FastText .vec files have the same format (first line is a header).
    """
    fasttext_path = Path(fasttext_path)
    if not fasttext_path.exists():
        raise FileNotFoundError(f"FastText file not found: {fasttext_path}")

    vocab_size = len(vocab) + 1
    matrix = np.random.normal(0.0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)
    matrix[0] = 0.0

    n_found = 0
    with open(fasttext_path, "r", encoding="utf-8") as fh:
        next(fh)                           # skip header line
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                try:
                    matrix[idx] = np.array(parts[1:], dtype=np.float32)
                    n_found += 1
                except ValueError:
                    pass

    coverage = n_found / max(len(vocab), 1) * 100
    logger.info(
        f"FastText coverage: {n_found:,}/{len(vocab):,} tokens ({coverage:.1f} %)"
    )
    return torch.tensor(matrix, dtype=torch.float32), n_found
