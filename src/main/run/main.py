import sys
sys.path.append("../scripts/helper/")
sys.path.append("../scripts/cnnmodels/")
from albumentations_script import *
from classmodules import *
from metrics import *
from attentionresnet34 import AttResNet34
from attentionunet import att_unet

import numpy as np
import pandas as pd
import gc
import keras

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split,StratifiedKFold

from skimage.transform import resize
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Concatenate, MaxPooling2D, UpSampling2D, Dropout
from tqdm import tqdm_notebook
from keras import initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU, ZeroPadding2D, multiply
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception


from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers import Activation, Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import add
from keras.layers import GlobalAveragePooling2D, Reshape, Permute
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil
import os
import random
from PIL import Image


seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

%matplotlib inline


train_im_path,train_mask_path = '../../resources/keras_im_train','../../resources/keras_mask_train'
val_im_path,val_mask_path = '../../resources/keras_im_val','../../resources/keras_mask_val'

h,w,batch_size = 256,256,16
epochs = 132

AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations_train(), get_augmentations_test()


class AttentionSWAModel():
    '''
    This class provides a common functionality for the SWA models.

    model_type represents whether it is a resnet based model / plain unet based model. Hence has only 2 values.
    epochs = number of epochs the model is to be trained for
    img_size = size of the image
    metric_list = a list of metrics to be monitored during the training process
    augmentations_train = augmentations included for the training set
    augmentations_test = augmentations included for the test set
    swa_epoch = epoch number post which stochastic weighted averaging is done
    train_img_path = path for training images
    train_mask_path = path for masks of the training images
    valid_img_path = path for validation images
    valid_mask_path = path for masks of the validation images
    snapshots = number of snapshots to be taken throughout the training process
    loss_func = loss function to be evaluated during training
    optimizer = optimization algorithm to be used during training
    init_lr = initial learning rate provided for the model

    '''
    def __init__(self, model_type, epochs, img_size, metric_list, augmentations_train, augmentations_test, swa_epoch, train_img_path, train_mask_path, valid_img_path, valid_mask_path, snapshots, loss_func, optimizer = 'adam', init_lr = 1e-4):
        self.epochs = epochs
        self.model_type = model_type
        self.img_size = img_size
        self.metric_list = metric_list
        self.augmentations_train = augmentations_train
        self.augmentations_test = augmentations_test
        self.swa_epoch = swa_epoch
        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.valid_img_path = valid_img_path
        self.valid_mask_path = valid_mask_path
        self.snapshots = snapshots
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.init_lr = init_lr
        if model_type.lower == 'plain':
            self.model_save_path = 'attunet_swa.h5'
            self.swa_checkpoint_path = 'keras_attunet_swa.model'
        else:
            self.model_save_path = 'attresnet34_swa.h5'
            self.swa_checkpoint_path = 'keras_attresnet34_swa.model'


    def train_model():
        global swa
        if self.model_type=='plain':
            seg_model = att_unet(256, 256, 1)
        else:
            seg_model = AttResNet34(input_shape=(256,256,3),encoder_weights=True)
        seg_model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=self.metric_list)
        snapshot = SnapshotCallbackBuilder(nb_epochs=self.epochs,nb_snapshots=self.snapshots,checkpoint_path = '../output/checkpoint/' + self.swa_checkpoint_path.split('.')[0] + '_checkpoint.h5', init_lr=self.init_lr)
        swa = get_swa('../output/checkpoint/' + self.swa_checkpoint_path, self.swa_epoch)
        training_generator = DataGenerator(augmentations=self.augmentations_train,img_size=self.img_size)
        validation_generator = DataGenerator(train_im_path = self.valid_img_path , train_mask_path = self.valid_mask_path ,augmentations=self.augmentations_test, img_size=self.img_size)
        print("Starting the training for " + self.model_save_path.split('.')[0])
        history = seg_model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,verbose=1, callbacks=snapshot.get_callbacks())
        seg_model.save('../output/models/' + self.model_save_path)
        print(self.model_save_path.split('.')[0] + ' saved in ' + '../output/models/' + self.model_save_path)
        return seg_model, history





class AttentionModel():
    '''
    This class provides a common functionality for the models which do not use SWA.

    model_type represents whether it is a resnet based model / plain unet based model. Hence has only 2 values.
    epochs = number of epochs the model is to be trained for
    img_size = size of the image
    metric_list = a list of metrics to be monitored during the training process
    augmentations_train = augmentations included for the training set
    augmentations_test = augmentations included for the test set
    train_img_path = path for training images
    train_mask_path = path for masks of the training images
    valid_img_path = path for validation images
    valid_mask_path = path for masks of the validation images
    loss_func = loss function to be evaluated during training
    init_lr = initial learning rate provided for the model

    '''
    def __init__(self, model_type, epochs, img_size, metric_list, augmentations_train, augmentations_test, train_img_path, train_mask_path, valid_img_path, valid_mask_path, loss_func, init_lr = 1e-4):
        self.epochs = epochs
        self.model_type = model_type
        self.img_size = img_size
        self.metric_list = metric_list
        self.augmentations_train = augmentations_train
        self.augmentations_test = augmentations_test
        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.valid_img_path = valid_img_path
        self.valid_mask_path = valid_mask_path
        self.loss_func = loss_func
        self.init_lr = init_lr
        if model_type.lower == 'plain':
            self.model_save_path = 'attunet.h5'
            self.model_checkpoint_path = 'keras_attunet_checkpoint.h5'
        else:
            self.model_save_path = 'attresnet.h5'
            self.model_checkpoint_path = 'keras_attresnet_checkpoint.h5'


    def train_model(factor=0.2, patience=1, mode='min', verbose=1, min_delta=0.0001, cooldown=2, min_lr=1e-7):
        global swa
        if self.model_type=='plain':
            seg_model = att_unet(256, 256, 1)
        else:
            seg_model = AttResNet34(input_shape=(256,256,3),encoder_weights=True)
        seg_model.compile(optimizer=Adam(learning_rate=self.init_lr), loss=self.loss_func, metrics=self.metric_list)
        checkpoint = ModelCheckpoint('../output/checkpoint/' + self.model_checkpoint_path, monitor='val_my_iou_metric', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, verbose=verbose, mode=mode, min_delta=min_delta, cooldown=cooldown, min_lr=min_lr)
        callbacks_list = [checkpoint, reduceLROnPlat]
        training_generator = DataGenerator(augmentations=self.augmentations_train,img_size=self.img_size)
        validation_generator = DataGenerator(train_im_path = self.valid_img_path , train_mask_path = self.valid_mask_path ,augmentations=self.augmentations_test, img_size=self.img_size)
        print("Starting the training for " + self.model_save_path.split('.')[0])
        history = seg_model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,verbose=1, callbacks=callbacks_list)
        seg_model.save('../output/models/' + self.model_save_path)
        print(self.model_save_path.split('.')[0] + ' saved in ' + '../output/models/' + self.model_save_path)
        return seg_model, history

# metric_list = [dice_coef, 'binary_accuracy', my_iou_metric]
# attresnet_swa =  AttentionSWAModel(model_type='resnet', epochs = epochs, img_size = h, metric_list = metric_list, augmentations_train=AUGMENTATIONS_TRAIN, augmentations_test=AUGMENTATIONS_TEST, swa_epoch=117, train_img_path=train_im_path, train_mask_path=train_mask_path, valid_img_path=val_im_path, valid_mask_path=val_mask_path, snapshots=4, loss_func=bce_dice_loss)
# seg_model, history = attresnet_swa.train_model()
#
# attunet_swa =  AttentionSWAModel(model_type='plain', epochs = epochs, img_size = h, metric_list = metric_list, augmentations_train=AUGMENTATIONS_TRAIN, augmentations_test=AUGMENTATIONS_TEST, swa_epoch=117, train_img_path=train_im_path, train_mask_path=train_mask_path, valid_img_path=val_im_path, valid_mask_path=val_mask_path, snapshots=4, loss_func=bce_dice_loss)
# seg_model, history = attunet_swa.train_model()
#
# attresnet = AttentionModel(model_type='resnet', epochs = epochs, img_size = h, metric_list = metric_list, augmentations_train=AUGMENTATIONS_TRAIN, augmentations_test=AUGMENTATIONS_TEST, train_img_path=train_im_path, train_mask_path=train_mask_path, valid_img_path=val_im_path, valid_mask_path=val_mask_path, loss_func=bce_dice_loss)
# seg_model, history = attresnet.train_model()
#
# attunet = AttentionModel(model_type='plain', epochs = epochs, img_size = h, metric_list = metric_list, augmentations_train=AUGMENTATIONS_TRAIN, augmentations_test=AUGMENTATIONS_TEST, train_img_path=train_im_path, train_mask_path=train_mask_path, valid_img_path=val_im_path, valid_mask_path=val_mask_path, loss_func=bce_dice_loss)
# seg_model, history = attunet.train_model()
