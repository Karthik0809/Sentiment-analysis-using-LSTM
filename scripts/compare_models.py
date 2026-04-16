#!/usr/bin/env python3
"""
Train all three architectures and produce a comparison report.

Usage
-----
python scripts/compare_models.py
"""
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
from src.utils.visualization import plot_model_comparison, plot_training_history

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("compare")


def main():
    config = load_config("configs/config.yaml")
    device = get_device()

    preprocessor = TextPreprocessor(
        max_len=config["data"]["max_len"],
        vocab_size=config["data"]["vocab_size"],
    )
    data_module = SentimentDataModule(config, preprocessor)

    data_path = "data/News_Category_Dataset_v3.json"
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    train_ds, val_ds, test_ds = data_module.load_and_prepare(data_path)
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch)
    test_loader  = DataLoader(test_ds,  batch_size=batch)

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/preprocessor.pkl", "wb") as fh:
        pickle.dump(preprocessor, fh)

    common = dict(
        vocab_size=preprocessor.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
    )

    experiments = {
        "Baseline LSTM":        BaselineLSTM(**common),
        "Stacked BiGRU":        StackedBiGRU(**common),
        "BiLSTM + Attention":   BiLSTMWithAttention(**common),
    }

    all_results = {}
    all_histories = {}

    for name, model in experiments.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*60}")
        ckpt_name = name.lower().replace(" ", "_").replace("+", "").replace("__", "_") + ".pt"
        trainer  = Trainer(model, config, device, checkpoint_dir="checkpoints")
        history  = trainer.train(train_loader, val_loader,
                                 epochs=config["training"]["epochs"])
        trainer.load(ckpt_name.replace(".pt", "") + ".pt" if os.path.exists(
            f"checkpoints/{ckpt_name}") else "best_model.pt")
        ev       = Evaluator(model, device)
        results  = ev.evaluate(test_loader)
        all_results[name]  = results
        all_histories[name] = history

        logger.info(f"{name}: Acc={results['accuracy']:.4f} F1={results['f1_macro']:.4f}")

    # ── Comparison plot ───────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    fig_cmp = plot_model_comparison(all_results)
    fig_cmp.write_html("outputs/model_comparison.html")

    for name, hist in all_histories.items():
        fig = plot_training_history(hist, model_name=name)
        safe = name.lower().replace(" ", "_").replace("+", "")
        fig.write_html(f"outputs/{safe}_history.html")

    logger.info("\nComparison complete. Plots saved to outputs/")
    logger.info("\nFinal Results:")
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'ROC-AUC':>10}")
    logger.info("-" * 60)
    for name, r in all_results.items():
        roc = r.get("roc_auc") or 0.0
        logger.info(f"{name:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {roc:>10.4f}")


if __name__ == "__main__":
    main()
