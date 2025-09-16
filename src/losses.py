# src/losses.py
import keras
import tensorflow as tf

from src import config


@tf.function
def dice_coef(y_true, y_pred, eps: float = 1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num = 2.0 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    den = tf.add(tf.reduce_sum(tf.add(y_true, y_pred), axis=[1, 2]), eps)
    dice = (num + eps) / den
    return tf.reduce_mean(dice, axis=-1)


def dice_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    if dice is None:
        return tf.constant(1.0, dtype=tf.float32)
    return 1.0 - dice


def bce_dice_loss(y_true, y_pred, bce_w: float = 0.5):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)  # [B,H,W,C]
    bce = tf.reduce_mean(bce, axis=[1, 2])  # [B,C] -> [B,]
    return tf.add(
        tf.multiply(tf.constant(bce_w, dtype=bce.dtype), bce),
        tf.multiply(
            tf.constant(1.0 - bce_w, dtype=bce.dtype), dice_loss(y_true, y_pred)
        ),
    )


class PerClassIoU(keras.metrics.Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "per_class_iou",
        n_classes: int = config.N_CLASSES,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.th = threshold
        self.n_classes = n_classes
        self.intersection = self.add_weight(
            shape=(n_classes,), initializer="zeros", name="int"
        )
        self.union = self.add_weight(
            shape=(n_classes,), initializer="zeros", name="uni"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.th, tf.float32)
        axes = [1, 2]
        inter = tf.reduce_sum(y_true * y_pred, axis=axes)  # [B,C]
        uni = tf.reduce_sum(y_true + y_pred, axis=axes) - inter  # [B,C]
        self.intersection.assign_add(tf.reduce_sum(inter, axis=0))
        self.union.assign_add(tf.reduce_sum(uni, axis=0))

    def result(self):
        eps = 1e-7
        return (self.intersection + eps) / (self.union + eps)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


def mean_iou_from_vector(vec):
    return tf.reduce_mean(vec)
