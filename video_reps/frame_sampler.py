"""Frame sampling strategies for video analysis."""

import cv2
import numpy as np


def uniform_sample(frames: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Evenly sample k frames across the full video.

    Args:
        frames: All video frames.
        k: Number of frames to sample.

    Returns:
        List of k evenly spaced frames.
    """
    n = len(frames)
    if k >= n:
        return list(frames)

    indices = np.linspace(0, n - 1, k, dtype=int)
    return [frames[i] for i in indices]


def motion_sample(frames: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Sample frames with highest motion scores, returned in temporal order.

    Uses grayscale frame-to-frame absolute difference to score motion.

    Args:
        frames: All video frames.
        k: Number of frames to sample.

    Returns:
        List of k frames with highest motion, in chronological order.
    """
    n = len(frames)
    if k >= n:
        return list(frames)

    # Convert to grayscale and compute motion scores
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    scores = np.zeros(n)
    for i in range(1, n):
        diff = cv2.absdiff(grays[i - 1], grays[i])
        scores[i] = np.mean(diff)

    # Select top-k indices by motion score, then sort chronologically
    top_indices = np.argsort(scores)[-k:]
    top_indices = np.sort(top_indices)

    return [frames[i] for i in top_indices]
