# src/viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .config import PALETTE

PALETTE_ARR = np.array(PALETTE, dtype=np.uint8)


def overlay(img: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.4):
    """Alpha-blend multi-hot mask over image using a fixed palette.
    img: [H,W,3] float 0..1
    mask_bin: [H,W,C] {0,1}
    """
    color = (mask_bin @ PALETTE_ARR).astype(np.float32) / 255.0
    blend = (1 - alpha) * img + alpha * color
    return np.clip(blend, 0, 1)


def show_triplet(img, gt, pred, title: str = ""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Input")
    axs[0].axis("off")
    axs[1].imshow(overlay(img, (gt > 0.5).astype(np.float32)))
    axs[1].set_title("GT")
    axs[1].axis("off")
    axs[2].imshow(overlay(img, (pred > 0.5).astype(np.float32)))
    axs[2].set_title("Pred")
    axs[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
