from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (256 x 256)
    s2 = vgg16.get_layer("block2_conv2").output         ## (128 x 128)
    s3 = vgg16.get_layer("block3_conv3").output         ## (64 x 64)
    s4 = vgg16.get_layer("block4_conv3").output         ## (32 x 32)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output         ## (16 x 16)

    """ Decoder """
    d1 = decoder_block(b1, s4, 256)                     ## (32 x 32)
    d2 = decoder_block(d1, s3, 128)                     ## (64 x 64)
    d3 = decoder_block(d2, s2, 64)                      ## (128 x 128)
    d4 = decoder_block(d3, s1, 32)                      ## (256 x 256)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_vgg16_unet(input_shape)
    model.summary()