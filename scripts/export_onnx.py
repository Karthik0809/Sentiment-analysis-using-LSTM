#!/usr/bin/env python3
"""
Export the trained PyTorch model to ONNX format for fast CPU inference.

ONNX Runtime is typically 2–4× faster than PyTorch CPU inference and
enables deployment in environments without a Python/PyTorch runtime
(Java, C++, .NET, mobile).

Usage
-----
python scripts/export_onnx.py
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/model.onnx

Verify / benchmark:
    python scripts/export_onnx.py --verify --benchmark
"""
import argparse
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.lstm import BiLSTMWithAttention
from src.utils.config import get_device, load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("export_onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--output",     default="checkpoints/model.onnx")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--seq-len",    type=int, default=50,
                   help="Sequence length used during export (must match training max_len)")
    p.add_argument("--verify",     action="store_true",
                   help="Verify ONNX output matches PyTorch output")
    p.add_argument("--benchmark",  action="store_true",
                   help="Benchmark ONNX vs PyTorch CPU throughput")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for benchmark (default: 1)")
    return p.parse_args()


def load_model(checkpoint_path: str, config: dict) -> tuple:
    """Load the PyTorch model from a checkpoint."""
    import pickle
    with open("checkpoints/preprocessor.pkl", "rb") as fh:
        preprocessor = pickle.load(fh)

    model = BiLSTMWithAttention(
        vocab_size=preprocessor.actual_vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        n_layers=config["model"]["n_layers"],
        dropout=0.0,                      # no dropout at export
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, preprocessor


def export(model: torch.nn.Module, seq_len: int, output_path: str) -> None:
    """Trace and export the model to ONNX."""
    dummy = torch.zeros(1, seq_len, dtype=torch.long)

    logger.info(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / 1_048_576
    logger.info(f"ONNX model saved: {output_path}  ({size_mb:.1f} MB)")


def verify(model: torch.nn.Module, onnx_path: str, seq_len: int) -> None:
    """Compare ONNX Runtime outputs to PyTorch outputs on random inputs."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime not installed — skipping verification. "
                       "Install with: pip install onnxruntime")
        return

    dummy_np = np.random.randint(0, 100, size=(1, seq_len), dtype=np.int64)
    dummy_pt = torch.tensor(dummy_np, dtype=torch.long)

    with torch.no_grad():
        pt_logits, _ = model(dummy_pt)
        pt_probs = torch.softmax(pt_logits, dim=1).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input_ids": dummy_np})
    ort_logits = ort_out[0]
    import numpy as np
    ort_probs = np.exp(ort_logits) / np.exp(ort_logits).sum(axis=-1, keepdims=True)

    max_diff = abs(pt_probs - ort_probs).max()
    if max_diff < 1e-4:
        logger.info(f"Verification PASSED — max output diff: {max_diff:.6f}")
    else:
        logger.warning(f"Verification WARNING — max output diff: {max_diff:.6f}")


def benchmark(model: torch.nn.Module, onnx_path: str, seq_len: int,
              batch_size: int, n_runs: int = 200) -> None:
    """Compare throughput: PyTorch CPU vs ONNX Runtime CPU."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime not installed — skipping benchmark.")
        return

    dummy_np = np.random.randint(0, 100, size=(batch_size, seq_len), dtype=np.int64)
    dummy_pt = torch.tensor(dummy_np, dtype=torch.long)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_pt)
        sess.run(["logits"], {"input_ids": dummy_np})

    # PyTorch
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy_pt)
    pt_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ONNX Runtime
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(["logits"], {"input_ids": dummy_np})
    ort_ms = (time.perf_counter() - t0) / n_runs * 1000

    speedup = pt_ms / ort_ms
    logger.info(
        f"\nBenchmark (batch={batch_size}, seq_len={seq_len}, n={n_runs})"
        f"\n  PyTorch CPU : {pt_ms:.2f} ms/batch"
        f"\n  ONNX Runtime: {ort_ms:.2f} ms/batch"
        f"\n  Speedup     : {speedup:.2f}×"
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists("checkpoints/preprocessor.pkl"):
        logger.error("Preprocessor not found. Run scripts/train.py first.")
        sys.exit(1)

    model, preprocessor = load_model(args.checkpoint, config)
    export(model, args.seq_len, args.output)

    if args.verify:
        verify(model, args.output, args.seq_len)

    if args.benchmark:
        benchmark(model, args.output, args.seq_len, args.batch_size)


if __name__ == "__main__":
    main()
