"""Preprocess video frames for model input."""

import cv2
import numpy as np
from PIL import Image


def preprocess_frames(
    frames: list[np.ndarray], size: int = 448
) -> list[Image.Image]:
    """Convert and resize BGR frames to PIL RGB images.

    Args:
        frames: List of BGR numpy frames from OpenCV.
        size: Target size (width and height) for resizing.

    Returns:
        List of PIL Images in RGB format, resized to size x size.
    """
    images = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        images.append(Image.fromarray(rgb))
    return images
