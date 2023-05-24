import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, Conv2D,MaxPooling1D, BatchNormalization, Input
from tensorflow.keras.layers import UpSampling1D, LeakyReLU, Conv1DTranspose, Concatenate, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1DTranspose
import math
from tensorflow.keras.initializers import RandomNormal


def upsampling_model(image_shape,upscale_factor=3):
    input_img = Input(shape=image_shape)
    conv_args = {
        "activation": "relu",
        "padding": "same",
        "strides": 1,
        "kernel_size":3
    }
    layers = int(math.log2(upscale_factor))
    x = Conv1D(filters = 64, **conv_args)(input_img)
    for i in range(layers):
        x = Conv1D(filters = 64, **conv_args)(x)
        x = UpSampling1D(2)(x)
    output_img = Conv1D(2, padding="same", strides=1, kernel_size = 3)(x)
    model = Model(input_img, output_img)
    return model


def transpose_model(image_shape,upscale_factor=3):
    input_img = Input(shape=image_shape)
    layers = int(math.log2(upscale_factor))
    x = Conv1D(filters = 64, activation="relu",padding="same",kernel_size=3)(input_img)
    for i in range(layers):
        #x = Conv1DTranspose(8*(i+1), kernel_size=3,strides=2, activation='relu', padding='same')(x)
        x = Conv1DTranspose(64, kernel_size=3,strides=2, activation='relu', padding='same')(x)
    output_img = Conv1D(2, 3, padding='same')(x)
    model = Model(input_img, output_img)
    return model

