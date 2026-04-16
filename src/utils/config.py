"""Configuration loading and device selection utilities."""
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    return config


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device
