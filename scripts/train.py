#!/usr/bin/env python3
"""
Training script for sentiment analysis models.

Usage
-----
# Train the default BiLSTM+Attention model
python scripts/train.py

# Train a specific architecture
python scripts/train.py --model bigru --epochs 20

# Custom config / data path
python scripts/train.py --config configs/config.yaml --data data/News_Category_Dataset_v3.json
"""
import argparse
import logging
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import SentimentDataModule
from src.data.preprocessor import TextPreprocessor
from src.models.lstm import BaselineLSTM, BiLSTMWithAttention, StackedBiGRU
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import get_device, load_config
from src.utils.embeddings import load_glove
from src.utils.visualization import plot_confusion_matrix, plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

MODEL_REGISTRY = {
    "bilstm":  BiLSTMWithAttention,
    "lstm":    BaselineLSTM,
    "bigru":   StackedBiGRU,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a news sentiment classifier")
    p.add_argument("--model",  default="bilstm", choices=list(MODEL_REGISTRY))
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data",   default="data/News_Category_Dataset_v3.json")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--glove-path", type=str, default=None,
                   help="Path to GloVe .txt file (e.g. data/glove/glove.6B.100d.txt). "
                        "Run scripts/download_glove.py first.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    # Allow CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # ── Data ─────────────────────────────────────────────────────────
    preprocessor = TextPreprocessor(
        max_len=config["data"]["max_len"],
        vocab_size=config["data"]["vocab_size"],
    )
    data_module = SentimentDataModule(config, preprocessor)

    if not os.path.exists(args.data):
        logger.error(
            f"Dataset not found at '{args.data}'.\n"
            "Download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset\n"
            "and place the JSON file at data/News_Category_Dataset_v3.json"
        )
        sys.exit(1)

    train_ds, val_ds, test_ds = data_module.load_and_prepare(args.data)

    batch = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0)

    # Save preprocessor for inference
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/preprocessor.pkl", "wb") as fh:
        pickle.dump(preprocessor, fh)
    logger.info("Preprocessor saved to checkpoints/preprocessor.pkl")

    # ── Pre-trained embeddings (optional) ────────────────────────────
    pretrained_emb = None
    if args.glove_path:
        pretrained_emb, n_found = load_glove(
            args.glove_path,
            preprocessor.vocab,
            config["model"]["embedding_dim"],
        )
        logger.info(f"Using GloVe embeddings: {n_found:,} tokens covered")

    # ── Model ─────────────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        vocab_size=preprocessor.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        pretrained_embeddings=pretrained_emb,
    )

    emb_label = f"GloVe ({args.glove_path})" if pretrained_emb is not None else "random init"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Embeddings: {emb_label} | Trainable params: {n_params:,}")

    # ── Training ──────────────────────────────────────────────────────
    trainer = Trainer(model, config, device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=config["training"]["epochs"],
    )

    # ── Test evaluation ───────────────────────────────────────────────
    trainer.load("best_model.pt")
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate(test_loader)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy   : {results['accuracy']:.4f}")
    logger.info(f"F1 (macro) : {results['f1_macro']:.4f}")
    logger.info(f"ROC-AUC    : {results.get('roc_auc') or 'N/A'}")
    logger.info("\nClassification Report:")
    logger.info(results["classification_report"])

    # ── Save plots ────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    label = args.model.upper()

    fig_hist = plot_training_history(history, model_name=label)
    fig_hist.write_html(f"outputs/{args.model}_history.html")

    fig_cm = plot_confusion_matrix(
        results["confusion_matrix"], ["Negative", "Neutral", "Positive"]
    )
    fig_cm.write_html(f"outputs/{args.model}_confusion_matrix.html")

    logger.info(f"Plots saved: outputs/{args.model}_*.html")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
