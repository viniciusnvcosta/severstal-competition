# src/model.py
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

# import torch
# import torch.nn as nn

from .config import IMG_H, IMG_W, N_CLASSES


def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def encoder_block(x, f):
    c = conv_block(x, f)
    p = layers.MaxPooling2D()(c)
    return c, p


def build_unet_light(input_shape=(IMG_H, IMG_W, 3), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, 16)
    c2, p2 = encoder_block(p1, 32)
    c3, p3 = encoder_block(p2, 64)
    c4, p4 = encoder_block(p3, 128)
    bn = conv_block(p4, 256)

    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 128)
    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 64)
    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 32)
    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 16)
    outputs = layers.Conv2D(n_classes, 1, activation="sigmoid")(c8)
    return keras.Model(inputs, outputs, name="unet_light")


#! NOT USED DUE TO RESTRICTION, JUST FOR REFERENCE
# def build_unet_effnetb0(input_shape=(IMG_H, IMG_W, 3), n_classes=N_CLASSES):
#     base = keras.applications.EfficientNetB0(
#         include_top=False, weights="imagenet", input_shape=input_shape
#     )
#     skip_names = [
#         "block2a_expand_activation",
#         "block3a_expand_activation",
#         "block4a_expand_activation",
#         "block6a_expand_activation",
#     ]
#     skips = [base.get_layer(n).output for n in skip_names]
#     x = base.output
#     for s in reversed(skips):
#         x = layers.UpSampling2D()(x)
#         x = layers.Concatenate()([x, s])
#         x = conv_block(x, 128)
#     x = layers.UpSampling2D(size=2)(x)
#     x = conv_block(x, 64)
#     outputs = layers.Conv2D(n_classes, 1, activation="sigmoid")(x)
#     return keras.Model(base.input, outputs, name="unet_effnetb0")


#* PyTorch version for reference
# class DoubleConv(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class UNetScratch(nn.Module):
#     def __init__(self, n_classes=4, base=16):
#         super().__init__()
#         self.c1 = DoubleConv(3, base)
#         self.p1 = nn.MaxPool2d(2)
#         self.c2 = DoubleConv(base, base * 2)
#         self.p2 = nn.MaxPool2d(2)
#         self.c3 = DoubleConv(base * 2, base * 4)
#         self.p3 = nn.MaxPool2d(2)
#         self.c4 = DoubleConv(base * 4, base * 8)
#         self.p4 = nn.MaxPool2d(2)
#         self.bn = DoubleConv(base * 8, base * 16)
#         self.u4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.d4 = DoubleConv(base * 16 + base * 8, base * 8)
#         self.u3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.d3 = DoubleConv(base * 8 + base * 4, base * 4)
#         self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.d2 = DoubleConv(base * 4 + base * 2, base * 2)
#         self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.d1 = DoubleConv(base * 2 + base, base)
#         self.out = nn.Conv2d(base, n_classes, 1)

#     def forward(self, x):
#         # Softmax is inappropriate because classes can co-occur per pixel.
#         c1 = self.c1(x)
#         p1 = self.p1(c1)
#         c2 = self.c2(p1)
#         p2 = self.p2(c2)
#         c3 = self.c3(p2)
#         p3 = self.p3(c3)
#         c4 = self.c4(p3)
#         p4 = self.p4(c4)
#         x = self.bn(p4)
#         x = self.u4(x)
#         x = torch.cat([x, c4], 1)
#         x = self.d4(x)
#         x = self.u3(x)
#         x = torch.cat([x, c3], 1)
#         x = self.d3(x)
#         x = self.u2(x)
#         x = torch.cat([x, c2], 1)
#         x = self.d2(x)
#         x = self.u1(x)
#         x = torch.cat([x, c1], 1)
#         x = self.d1(x)
#         return self.out(x)  # logits
