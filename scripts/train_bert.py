#!/usr/bin/env python3
"""
Fine-tune DistilBERT for 3-class news headline sentiment classification.

Usage
-----
python scripts/train_bert.py                           # 5 epochs, batch=32, lr=2e-5
python scripts/train_bert.py --epochs 3 --batch-size 64
python scripts/train_bert.py --output-dir checkpoints/distilbert
"""
import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.bert_dataset import BertDataModule
from src.data.preprocessor import TextPreprocessor
from src.models.transformer import DistilBertSentiment
from src.utils.config import get_device, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_bert")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for news sentiment")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--data",        default="data/News_Category_Dataset_v3.json")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--max-len",     type=int,   default=64,
                   help="Max token length (headlines are short; 64 is sufficient)")
    p.add_argument("--output-dir",  default="checkpoints/distilbert",
                   help="Where to save the best model and tokenizer")
    return p.parse_args()


def evaluate_loader(model, loader, device):
    """Return (accuracy, avg_loss) on a DataLoader."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(ids, mask)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total, total_loss / len(loader)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    device = get_device()
    logger.info(f"Device: {device}")

    # ── Tokenizer + data ──────────────────────────────────────────────
    tokenizer    = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    preprocessor = TextPreprocessor()          # only needed for get_label()

    data_module = BertDataModule(config, tokenizer, preprocessor)
    train_ds, val_ds, test_ds = data_module.load_and_prepare(
        args.data, max_len=args.max_len
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,       shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,   shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,   shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────
    model = DistilBertSentiment(num_labels=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DistilBERT trainable parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = torch.nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_acc = 0.0
    history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_acc, val_loss = evaluate_loader(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"  --> Best model saved  (val_acc={val_acc:.4f})")

    # ── Final test evaluation (best checkpoint) ───────────────────────
    best_model = DistilBertSentiment.from_pretrained(args.output_dir).to(device)
    test_acc, test_loss = evaluate_loader(best_model, test_loader, device)

    logger.info("=" * 60)
    logger.info(f"Best val_acc  : {best_val_acc:.4f}")
    logger.info(f"Test accuracy : {test_acc:.4f}")
    logger.info(f"Test loss     : {test_loss:.4f}")
    logger.info(f"Model saved → : {args.output_dir}/")

    history["best_val_acc"] = best_val_acc
    history["test_acc"]     = test_acc
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
