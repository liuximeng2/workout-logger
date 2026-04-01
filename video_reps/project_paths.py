"""Pin cache directories under the repository root (not ~/.cache)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_local_caches() -> None:
    """Set env vars before huggingface_hub / transformers / pip use default caches."""
    cache = _REPO_ROOT / ".cache"
    hf = cache / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
