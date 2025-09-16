# src/augment.py
import tensorflow as tf


def augment_basic(img, mask, seed: int = 42):
    """Apply consistent geometric + mild photometric augs to image & mask."""
    # One shared coin for both image and mask
    coin = tf.random.uniform([], 0, 1, seed=seed)

    def _flip(x):
        return tf.image.flip_left_right(x)

    img = tf.cond(coin < 0.5, lambda: _flip(img), lambda: img)
    mask = tf.cond(coin < 0.5, lambda: _flip(mask), lambda: mask)
    # Photometric (image only)
    img = tf.image.random_brightness(img, 0.05, seed=seed)
    img = tf.image.random_contrast(img, 0.9, 1.1, seed=seed)
    return img, mask
