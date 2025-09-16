# src/config.py
# Centralized knobs so you can tweak once and reuse everywhere.
IMAGES_DIR = 'data/train_images'
CSV_PATH = 'data/train.csv'

SEED: int = 42
IMG_H: int = 256
IMG_W: int = 1024 # originally 1600, reduced for faster training
N_CLASSES: int = 4
CLASS_NAMES = [f"class_{i}" for i in range(1, N_CLASSES + 1)]

# Consistent colors for overlays (R,G,B)
PALETTE = [
    [0, 255, 255],  # class 1 - cyan
    [255, 0, 255],  # class 2 - magenta
    [255, 255, 0],  # class 3 - yellow
    [50, 205, 50],  # class 4 - lime green
]
