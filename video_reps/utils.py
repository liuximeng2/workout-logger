"""Shared utilities for the video_reps package."""

import json
import logging
import re


def _parse_json_object(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_kv_fallback(text: str) -> dict | None:
    """Recover exercise/confidence/reps when the model returns broken JSON (common with VLMs)."""
    out: dict = {}
    m_ex = re.search(r'"exercise"\s*:\s*"([^"]*)"', text)
    if m_ex:
        name = m_ex.group(1).strip().rstrip(",").strip()
        out["exercise"] = name
    m_conf = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if m_conf:
        out["confidence"] = float(m_conf.group(1))
    m_reps = re.search(r'"reps"\s*:\s*([0-9]+)', text)
    if m_reps:
        out["reps"] = int(m_reps.group(1))
    return out if out else None


def extract_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown fences and extra text."""
    text = text.strip()
    parsed = _parse_json_object(text)
    if parsed is not None:
        return parsed

    # Markdown fence: use greedy brace span (non-greedy stops at first `}` and breaks nested objects).
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fence_match:
        parsed = _parse_json_object(fence_match.group(1))
        if parsed is not None:
            return parsed

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        parsed = _parse_json_object(obj_match.group(0))
        if parsed is not None:
            return parsed

    fallback = _extract_kv_fallback(text)
    if fallback is not None:
        return fallback

    raise ValueError(f"Could not extract valid JSON from model output: {text[:200]}")


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return a logger for the package."""
    logger = logging.getLogger("video_reps")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
