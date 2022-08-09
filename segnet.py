# -*- coding: utf-8 -*-
"""segnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_ukoA37skdU3MYtpvTZ1_BePKLQHWYN
"""

!pip install tensorflow_addons

import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils
from tensorflow_addons.layers import MaxUnpooling2D
from tensorflow.keras.applications import VGG16


def SegNet(input_shape, classes=1):
    img_input = Input(shape=input_shape)
    x = img_input

    # Encoder
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=img_input)

    """ Encoder """
    s1 = vgg16.get_layer("block1_pool").output         ## (256 x 256)
    s2 = vgg16.get_layer("block2_pool").output         ## (128 x 128)
    s3 = vgg16.get_layer("block3_pool").output         ## (64 x 64)
    s4 = vgg16.get_layer("block4_pool").output         ## (32 x 32)
    s5_mask = vgg16.get_layer("block5_conv3").output
    s5 = vgg16.get_layer("block5_pool").output         ## (16 x 16)
    
    # Decoder

    #block 1
    x = MaxUnpooling2D((3,3))(s5, s5_mask)

    x1 = Conv2D(512, 3, padding = "same")(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(512, 3, padding = "same")(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(512, 3, padding = "same")(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    #block 2

    x4 = MaxUnpooling2D((3,3))(s4, x3)
    
    x5 = Conv2D(256, 3, padding = "same")(x4)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)

    x6 = Conv2D(256, 3, padding = "same")(x5)
    x6 = BatchNormalization()(x6)
    x6 = Activation("relu")(x6)

    x7 = Conv2D(256, 3, padding = "same")(x6)
    x7 = BatchNormalization()(x7)
    x7 = Activation("relu")(x7)
    
    #block 3

    x8 = MaxUnpooling2D((3,3))(s3, x7)
    
    x9 = Conv2D(128, 3, padding = "same")(x8)
    x9 = BatchNormalization()(x9)
    x9 = Activation("relu")(x9)

    x10 = Conv2D(128, 3, padding = "same")(x9)
    x10 = BatchNormalization()(x10)
    x10 = Activation("relu")(x10)

    x11 = Conv2D(128, 3, padding = "same")(x10)
    x11 = BatchNormalization()(x11)
    x11 = Activation("relu")(x11)

    #block 4
    
    x12 = MaxUnpooling2D((2,2))(s2, x11)
    
    x13 = Conv2D(64, 2, padding = "same")(x12)
    x13 = BatchNormalization()(x13)
    x13 = Activation("relu")(x13)

    x14 =Conv2D(64, 2, padding = "same")(x13)
    x14 = BatchNormalization()(x14)
    x14 = Activation("relu")(x14)

    #block 5

    x15 = MaxUnpooling2D((2,2))(s1, x14)
    
    x16 = Conv2D(32, 3, padding = "same")(x15)
    x16 = BatchNormalization()(x16)
    x16 = Activation("relu")(x16)

    x17 = Conv2D(32, 3, padding = "same")(x16)
    x17 = BatchNormalization()(x17)
    x17 = Activation("relu")(x17)

    
    
    outputs = Conv2D(1, 1, padding="same", activation="softmax")(x17)
    model = Model(img_input, outputs)
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = SegNet(input_shape)
    model.summary()