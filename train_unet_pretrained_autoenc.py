from keras.preprocessing import image
from keras.applications import resnet50, densenet, mobilenet_v2
from keras.models import Model, model_from_json
from keras.layers import Reshape, Concatenate, Conv2D, Conv2DTranspose, Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.losses import cosine_proximity
from keras import regularizers
from PIL import Image
from random import randint
from unet_vanilla import unet
import numpy as np
import pandas as pd
import glob
import math
import warnings
import keras.backend as K
import pdb
import tensorflow as tf

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def read_imgs_keraspp(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 3))
    imgs_labels = np.zeros((len(imgs_df), 224, 224, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_tensor[index, :, :, :] = np.array(Image.open(row['image']).resize([224, 224]).convert('RGB'))
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([224, 224]))/255.0

    return imgs_tensor, imgs_labels

def batch_generator(train_df, batch_size=64):
    while True:
        for batch_index in range(round(len(train_df)/batch_size)):
            x_train, y_train = read_imgs_keraspp(train_df[batch_index*batch_size:batch_index*batch_size+batch_size])
            x_train = densenet.preprocess_input(x_train)

            yield x_train, y_train

def cosine_similarity(predictions, doc_embeddings):
    num = K.sum(predictions * doc_embeddings, axis=-1,keepdims=True)
    den = K.sqrt(K.sum(predictions * predictions, axis=-1, keepdims=True)) * K.sqrt(K.sum(doc_embeddings * doc_embeddings, axis=-1, keepdims=True))
    loss = -K.mean(num / den, axis=-1, keepdims=True)
    return loss

def L1_distance(merged_embeddings, dummy_vector):
    loss = K.mean(K.sum(K.abs(merged_embeddings - dummy_vector), axis=-1, keepdims=True))

    return loss

def iou(masks_true, masks_pred):
    """
    Get the IOU between each predicted mask and each true mask.

    Parameters
    ----------

    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    array-like
        A 2D array of shape (n_true_masks, n_predicted_masks), where
        the element at position (i, j) denotes the IoU between the `i`th true
        mask and the `j`th predicted mask.

    """
    if masks_true.shape[1:] != masks_pred.shape[1:]:
        raise ValueError('Predicted masks have wrong shape!')
    n_true_masks, height, width = masks_true.shape
    n_pred_masks = masks_pred.shape[0]
    m_true = masks_true.copy().reshape(n_true_masks, height * width).T
    m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
    numerator = np.dot(m_pred, m_true)
    denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)

    return numerator / (denominator - numerator)

def evaluate_image(masks_true, masks_pred, thresholds=0.5):
    """
    Get the average precision for the true and predicted masks of a single image,
    averaged over a set of thresholds

    Parameters
    ----------
    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    float
        The mean average precision of intersection over union between
        all pairs of true and predicted region masks.

    """
    int_o_un = iou(masks_true, masks_pred)
    benched = int_o_un > thresholds
    tp = benched.sum(-1).sum(-1)  # noqa
    fp = (benched.sum(2) == 0).sum(1)
    fn = (benched.sum(1) == 0).sum(1)

    return np.mean(tp / (tp + fp + fn))

def dice_coef(y_true, y_pred, smooth=1):
    y_pred_bool = K.greater(y_pred, 0.5)
    y_pred = K.cast(y_pred_bool, K.floatx())
    intersection = K.sum(K.abs(y_true * y_pred)) + 1e-4
    return (2.0 *intersection / (K.sum(y_true) + K.sum(y_pred) + 1e-4))

def learning_rate_scheduler(epoch):
    lr = 1e-4
    '''
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    '''
    print("Set Learning Rate : {}".format(lr))
    return lr

batch_size = 6

train_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_train_sentinel.csv')
val_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_val_sentinel.csv')

model = unet()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate_scheduler(0)),
              metrics=[dice_coef, 'acc'])

#filepath="weights-improvement-{epoch:02d}-{val_dice_coef:.2f}.hdf5"
filepath="best-unet-pretrained-gan.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('log_512_200_unet_pretrained_gan.csv', append=True, separator=';')
callbacks_list = [checkpoint, csv_logger]

model.fit_generator(batch_generator(train_df, batch_size), steps_per_epoch=round(len(train_df)/batch_size),
        epochs=80, validation_data=batch_generator(val_df, batch_size), validation_steps=round(len(val_df)/batch_size),callbacks=callbacks_list)
