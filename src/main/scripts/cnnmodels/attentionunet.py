import sys
sys.path.append("")

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, Add, Multiply, Concatenate, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
import keras.backend as K
from metrics import *
import tensorflow as tf
from keras.losses import binary_crossentropy

########################################################################################################
# Defining the attention block
########################################################################################################

def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    x = Add()([theta_x, phi_g])
    f = Activation('relu')(x)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = Multiply()([x, rate])
    return att_x


def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: Concatenate(axis=1)([x[0], x[1]]))
    else:
        my_concat = Lambda(lambda x: Concatenate(axis=3)([x[0], x[1]]))

    concate = my_concat([up, layer])
    return concate


########################################################################################################
# Defining the skeletal the neural network - UNET
########################################################################################################

def unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((img_h, img_w, 3))
    x = inputs
    depth = 5
    features = 16
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format= data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        x = Concatenate(axis=1)([skips[i], x])
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', activation = 'sigmoid', data_format=data_format)(x)
    model = Model(inputs=inputs, outputs=conv6)
    return model


########################################################################################################
#  Attention U-Net architecture
########################################################################################################
def att_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((img_h, img_w, 3))
    x = inputs
    depth = 5
    features = 16
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', activation='sigmoid', data_format=data_format)(x)
    model = Model(inputs=inputs, outputs=conv6)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef, 'binary_accuracy', true_positive_rate, my_iou_metric])
    return model
