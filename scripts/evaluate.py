#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the test set and print metrics.

Usage
-----
python scripts/evaluate.py
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --model bilstm
"""
import argparse
import json
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
from src.utils.config import get_device, load_config
from src.utils.visualization import plot_confusion_matrix, plot_model_comparison

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluate")

MODEL_REGISTRY = {
    "bilstm": BiLSTMWithAttention,
    "lstm":   BaselineLSTM,
    "bigru":  StackedBiGRU,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="bilstm", choices=list(MODEL_REGISTRY))
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--data",       default="data/News_Category_Dataset_v3.json")
    p.add_argument("--save-report", action="store_true",
                   help="Save JSON metrics to outputs/eval_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    # Load saved preprocessor
    prep_path = "checkpoints/preprocessor.pkl"
    if not os.path.exists(prep_path):
        logger.error("Preprocessor not found. Run scripts/train.py first.")
        sys.exit(1)
    with open(prep_path, "rb") as fh:
        preprocessor = pickle.load(fh)

    # Rebuild test split
    data_module = SentimentDataModule(config, preprocessor)
    _, _, test_ds = data_module.load_and_prepare(args.data)
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Load model
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        vocab_size=preprocessor.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=0.0,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    ev = Evaluator(model, device)
    results = ev.evaluate(test_loader)

    logger.info("\n" + "=" * 60)
    logger.info(f"Accuracy   : {results['accuracy']:.4f}")
    logger.info(f"F1 (macro) : {results['f1_macro']:.4f}")
    logger.info(f"F1 (weighted): {results['f1_weighted']:.4f}")
    logger.info(f"Precision  : {results['precision']:.4f}")
    logger.info(f"Recall     : {results['recall']:.4f}")
    logger.info(f"ROC-AUC    : {results.get('roc_auc') or 'N/A'}")
    logger.info("\n" + results["classification_report"])

    os.makedirs("outputs", exist_ok=True)
    fig = plot_confusion_matrix(
        results["confusion_matrix"], ["Negative", "Neutral", "Positive"]
    )
    fig.write_html("outputs/eval_confusion_matrix.html")

    if args.save_report:
        report = {
            k: v for k, v in results.items()
            if k not in ("predictions", "labels", "probabilities", "classification_report")
        }
        with open("outputs/eval_report.json", "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Report saved to outputs/eval_report.json")


if __name__ == "__main__":
    main()
