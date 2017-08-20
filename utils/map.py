"""Functions for mapping between different colormaps

Author: Michelle Guo
Date: August 2018"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import numpy as np


def grayscale_to_ocean(img, white=-1):
    """
    Converts a 1-channel grayscale image into an ocean image.
    Args:
        img (np.ndarray): (H, W) numpy array representing the grayscale image.
        white (int): Max grayscale threshold to use as white in the resulting image. To use image max, set to -1.
    Returns:
        img (np.ndarray): (H, W, 3) numpy array representing the colored grayscale image.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] not in [1, 3]:
        raise ValueError('Input image must have 1 or 3 channels.')

    img = img.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # float between 0, 255
    max_threshold = white if white >= 0 else np.max(img)
    img = np.clip(img, 0, max_threshold) / max_threshold
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    return img
