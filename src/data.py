# src/data.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import CLASS_NAMES, IMG_H, IMG_W, IMAGES_DIR, N_CLASSES

AUTOTUNE = tf.data.AUTOTUNE


# ----------------- RLE helpers -----------------
def decode_rle_to_mask(rle, height, width) -> np.ndarray:
    """Decode 1-indexed, column-major RLE to a binary mask [H, W]."""
    if isinstance(rle, bytes):
        rle = rle.decode("utf-8")
    if isinstance(rle, tf.Tensor):
        rle = rle.numpy()
        if isinstance(rle, bytes):
            rle = rle.decode("utf-8")
    if rle is None or rle == "" or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((height, width), np.uint8)
    s = np.asarray([int(x) for x in rle.strip().split()], dtype=np.int64)
    starts = s[0::2] - 1  # to 0-index
    lengths = s[1::2]
    ends = starts + lengths
    flat = np.zeros(height * width, dtype=np.uint8)
    for st, en in zip(starts, ends):
        flat[st:en] = 1
    return flat.reshape((height, width), order="F")  # column-major -> (H,W)


def encode_mask_to_rle(mask_2d: np.ndarray) -> str:
    """Encode a binary mask [H, W] to 1-indexed, column-major RLE."""
    h, w = mask_2d.shape
    flat = mask_2d.T.flatten(order="F")
    flat = np.concatenate([[0], flat, [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


# ----------------- CSV -> wide table -----------------
def build_image_table(csv_path: str, images_dir: str) -> pd.DataFrame:
    """Build a pivot table with one row per image and one column per class."""
    df = pd.read_csv(csv_path)
    assert set(["ImageId", "ClassId", "EncodedPixels"]).issubset(df.columns)
    wide = df.pivot_table(
        index="ImageId",
        columns="ClassId",
        values="EncodedPixels",
        aggfunc="first",
    ).reindex(columns=range(1, N_CLASSES + 1))
    wide.columns = [f"class_{c}" for c in wide.columns]
    # include images that have no rows in CSV (no-defect images)
    all_imgs = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    full = (
        pd.DataFrame({"ImageId": all_imgs}).set_index("ImageId").join(wide, how="left")
    )
    return full


def basic_counts(full: pd.DataFrame) -> Dict:
    n_images = len(full)
    per_class = {c: int(full[c].notna().sum()) for c in CLASS_NAMES}
    multi_def = int((full[CLASS_NAMES].notna().sum(axis=1) >= 2).sum())
    any_def = int((full[CLASS_NAMES].notna().sum(axis=1) >= 1).sum())
    none_def = n_images - any_def
    return {
        "n_images": n_images,
        "per_class_counts": per_class,
        "multi_defect_images": multi_def,
        "no_defect_images": none_def,
        "imbalance_ratio": {c: per_class[c] / max(1, any_def) for c in CLASS_NAMES},
    }


def records_from_table(full: pd.DataFrame, images_dir: str) -> List[Dict]:
    records = []
    for img_id, row in full.iterrows():
        records.append(
            {
                "path": os.path.join(images_dir, str(img_id)),
                **{c: ("" if pd.isna(row[c]) else str(row[c])) for c in CLASS_NAMES},
            }
        )
    return records


#! 2.4 REQUIREMENT


def name_and_mask(image_id: str, class_id: Optional[int] = None, full: Optional[pd.DataFrame] = None):
    """Return (image_id, image_np[H,W,3] float32 0..1, mask_np[H,W,4] float32 0/1)
    If class_id is None → composite of all classes; else one-hot in that channel.
    """

    path = os.path.join(IMAGES_DIR, image_id)
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32).numpy()
    h, w = img.shape[:2]
    masks = []
    for cid in range(1, 5):
        rle = (
            full.loc[image_id, f"class_{cid}"]
            if f"class_{cid}" in full.columns
            else None
        )
        m = (
            decode_rle_to_mask(rle, h, w)
            if pd.notna(rle)
            else np.zeros((h, w), np.uint8)
        )
        masks.append(m)
    mask4 = np.stack(masks, axis=-1).astype(np.float32)
    if class_id is not None:
        keep = np.zeros_like(mask4)
        keep[..., class_id - 1] = mask4[..., class_id - 1]
        mask4 = keep
    return image_id, img, mask4


# ----------------- tf.data pipeline -----------------
def _decode_img_and_mask(img_bytes, rles):
    img = tf.image.decode_jpeg(img_bytes)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    shape = tf.shape(img)
    h, w = shape[0], shape[1]

    def _decode_py(h_, w_, *rle_list):
        masks = []
        for r in rle_list:
            r = r.decode("utf-8") if isinstance(r, (bytes, bytearray)) else r
            masks.append(decode_rle_to_mask(r, int(h_), int(w_)))
        m = np.stack(masks, axis=-1).astype(np.float32)
        return m

    mask = tf.py_function(func=_decode_py, inp=[h, w, *rles], Tout=tf.float32)
    mask = tf.ensure_shape(mask, [None, None, N_CLASSES])

    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    mask = tf.image.resize(mask, (IMG_H, IMG_W), method="nearest")
    return img, mask


def make_dataset(
    records,
    batch=8,
    shuffle=True,
    augment_fn=None,
    seed=42,
    shuffle_buffer=256,
    nmap=4,
    prefetch_size=2,
):
    paths = [r["path"] for r in records]
    rle_cols = [[r[c] for r in records] for c in CLASS_NAMES]
    ds = tf.data.Dataset.from_tensor_slices((paths, *rle_cols))

    # 1) SHUFFLE FIRST (cheap: just filenames & RLE strings)
    if shuffle:
        ds = ds.shuffle(
            min(shuffle_buffer, len(paths)), seed=seed, reshuffle_each_iteration=True
        )

    # 2) MAP/DECODE with limited parallelism (py_function is CPU/Python-bound)
    @tf.autograph.experimental.do_not_convert
    def _load(path, *rles):
        img_bytes = tf.io.read_file(path)
        img, mask = _decode_img_and_mask(img_bytes, rles)  # your existing helper
        if augment_fn is not None:
            img, mask = augment_fn(img, mask, seed=seed)
        return img, mask

    ds = ds.map(_load, num_parallel_calls=nmap)

    # 3) BATCH THEN PREFETCH (small prefetch to avoid RAM spikes)
    ds = ds.batch(batch, drop_remainder=False).prefetch(prefetch_size)

    # 4) OK for training and can speed things up
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)
    return ds


def stratified_split(full: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = full.copy()
    df["any_defect"] = (df[CLASS_NAMES].notna().sum(axis=1) > 0).astype(int)
    idx = df.index.to_numpy()
    mask = df["any_defect"].to_numpy()
    pos = idx[mask == 1]
    neg = idx[mask == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    sp = int(train_ratio * len(pos))
    sn = int(train_ratio * len(neg))
    train_idx = set(pos[:sp]).union(set(neg[:sn]))
    val_idx = set(pos[sp:]).union(set(neg[sn:]))
    return train_idx, val_idx
