####################################################################################################################
# THIS SCRIPT IS TO BE USED AS A GENERAL SCRIPT FOR BOTH STEPS
####################################################################################################################

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


####################################################################################################################
# STEP 1: Get max IoU value for each model
####################################################################################################################
train_im_path,train_mask_path = '../../resources/keras_im_train','../../resources/keras_mask_train'
val_im_path,val_mask_path = '../../resources/keras_im_val','../../resources/keras_mask_val'

AUGMENTATIONS_TEST_FLIPPED = get_augmentations_test_flipped()

img_size = 256
validation_generator = DataGenerator(train_im_path = val_im_path, train_mask_path=val_mask_path,augmentations=AUGMENTATIONS_TEST, img_size=img_size,shuffle=False)
validation_generator_flipped = DataGenerator(train_im_path = val_im_path, train_mask_path=val_mask_path,augmentations=AUGMENTATIONS_TEST_FLIPPED, img_size=img_size,shuffle=False)

# uncomment the below lines and provide the model h5 file as per requirement (an example shown below)
# metric_list = [dice_coef, 'binary_accuracy', my_iou_metric]
# model_save_path = '../output/models/attresnet_swa.h5'
# seg_model = AttResNet34(input_shape=(256,256,3),encoder_weights=True)
# seg_model.compile(optimizer=Adam(learning_rate=1e-4), loss=bce_dice_loss, metrics=self.metric_list)
# seg_model.load_weights(model_save_path)

def predict_result(model,validation_generator,img_size):
    # TBD predict both orginal and reflect x
    preds_test1 = model.predict_generator(validation_generator).reshape(-1, img_size, img_size)
    return preds_test1

preds_valid_orig = predict_result(seg_model,validation_generator,img_size)
preds_valid_flipped = predict_result(seg_model,validation_generator_flipped,img_size)
preds_valid_flipped = np.array([np.fliplr(x) for x in preds_valid_flipped])
preds_valid = 0.5*preds_valid_orig + 0.5*preds_valid_flipped


valid_fn = glob.glob(val_mask_path)
y_valid_ori = np.array([cv2.resize(np.array(Image.open(fn)),(img_size,img_size)) for fn in valid_fn])
assert y_valid_ori.shape == preds_valid.shape

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

y_valid_ori = np.array([cv2.resize(np.array(Image.open(fn)),(img_size,img_size)) for fn in valid_fn])
assert y_valid_ori.shape == preds_valid.shape

thresholds = np.linspace(0.2, 0.9, 31)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]





############################################################################################################
# STEP 2: Use iou_best to add respective entries in model_info dictionary
############################################################################################################
def get_attunet():
	return att_unet(256, 256, 1)

def get_attresnet34():
	return AttResNet34(input_shape=(256,256,3),encoder_weights=False)


# model_info = {'<model_name (as per h5 file stored)> : [<base_network()>, iou_best]'}
#
# For example:
# model_info = {'attresnet34_swa' : [get_attresnet(), 0.9], 'attresnet34' : [get_attresnet(), 0.9],'attunet_swa' : [get_attunet(), 0.9], 'attunet' : [get_attunet(), 0.9]}


def pred_choice(img_path, batch_size=16, img_size=256):
	print('Starting prediction to compare between models')
	x_test = [cv2.resize(cv2.imread(img_path), (img_size, img_size))]
	x_test = np.array(x_test)
	print(x_test.shape)
	num_models = len(model_info)
	i = 0
	fig, axs = plt.subplots(1, num_models, figsize=(12, 6))
	img = x_test[0]
	for model_name in model_info.keys():
		seg_model = model_info[model_name][0]
		seg_model.load_weights('../models/' + model_name + '_30.h5')
		preds_test_orig = seg_model.predict(x_test,batch_size=batch_size)
		x_test = np.array([np.fliplr(x) for x in x_test])
		preds_test_flipped = seg_model.predict(x_test,batch_size=batch_size)
		preds_test_flipped = np.array([np.fliplr(x) for x in preds_test_flipped])
		preds_test = 0.5*preds_test_orig + 0.5*preds_test_flipped
		threshold_best = model_info[model_name][1]
		pred = preds_test.squeeze()
		ax = axs[i]
		ax.imshow(img, cmap="Greys")
		ax.imshow(np.array(np.round(pred > threshold_best), dtype=np.float32), alpha=0.5, cmap="Reds")
		ax.axis('off')
		ax.set_title(model_name)
		i += 1

	print('Saving image')
	fig.suptitle('Comparison between models')
	if os.path.exists("../../resources/saved_output/result.jpeg"):
  		os.remove("result.jpeg")
	plt.savefig("result.jpeg")

# Call the below function by providing the path to the test image to compare predictions between different models
# pred_choice('../../resources/input/test_img1.png')
