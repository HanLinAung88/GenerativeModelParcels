from keras.preprocessing import image
from keras.applications import resnet50, densenet, mobilenet_v2
from keras.models import Model, model_from_json
from keras.layers import Reshape, Concatenate, Conv2D, Conv2DTranspose, Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
#from segmentation_models.metrics import iou_score, dice_score
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.losses import cosine_proximity
from keras import regularizers
from PIL import Image
from random import randint
from unet_vanilla import unet
#from unet_sentinel import unet
import numpy as np
import pandas as pd
import glob
import math
import warnings
import keras.backend as K
import pdb
import tensorflow as tf
from keras.models import load_model
import keras.losses
from matplotlib import pyplot as plt
#from utils.metrics import get_metrics, f1, dice_coef_burak

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

#Set the filepaths here for laoding in the file 
is_fill = False
is_stacked = False
is_imageNet = False
image_type = 'sentinel' # OR 'Digital_Globe'



batch_size = 1
num_channels = 3
if is_stacked:
    num_channels = 9

if image_type == 'sentinel':
    input_shape = (224,224,num_channels)
else:
    input_shape = (512, 512, num_channels)

batch_size = 32
"""
base_dir = './data/' + image_type + '/'
val_file = 'parcel_segmentation_val_' + image_type
filepath= 'best-unet-' + image_type
csv_log_file = 'log_512_unet_' + image_type

sub_fill = ''
if is_fill:
    sub_fill = '_fill'

#Modify file path depending on fill/boundary task
train_file = train_file + sub_fill + '.csv'
val_file = val_file + sub_fill + '.csv'
filepath = filepath + sub_fill + '.hdf5'
csv_log_file = csv_log_file + sub_fill + '.csv'

#Loads validation data frame
test_df = pd.read_csv(base_dir + val_file)
pred_dir = "predictions/" + image_type + sub_fill  + '_' + num_channels + '_' + int(is_imageNet)

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_fname = pred_dir + "unet_predictions.npy"
"""
model = unet()
"""
if is_imageNet:
    model = Unet(BACKBONE, encoder_weights='imagenet')
else:
    model = unet(input_size=input_shape)
"""

def dice_coef(y_true, y_pred, smooth=1):
    y_pred_bool = K.greater(y_pred, 0.5)
    y_pred = K.cast(y_pred_bool, K.floatx())
    intersection = K.sum(K.abs(y_true * y_pred)) + 1e-4
    return (2.0 *intersection / (K.sum(y_true) + K.sum(y_pred) + 1e-4))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate_scheduler(0)),
              metrics=[dice_coef, 'acc'])

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
#model.compile(loss='binary_crossentropy',
#              optimizer=Adam(lr=learning_rate_scheduler(0)),
#              metrics=[dice_coef_burak, dice_score, iou_score, 'acc', f1])


filepath="beest/best-188-0.36.hdf5"
#filepath="best-unet-36.hdf5"
filepath="best-unet-dilated-rev.hdf5"

filepath="best-unet-sentinel.hdf5"

filepath="best-unet-sentinel-filled.hdf5"
filepath="best-unet-pretrained-gan.hdf5"
filepath="best-2-unet.hdf5"
pred_file="unet_pretrained_imagenet.npy"
#filepath="weights-improvement-{epoch:02d}-{val_dice_coef:.2f}.hdf5"
#filepath='weights-improvement-59-0.36.hdf5'
dependencies = {'dice_coef': dice_coef }
model = load_model(filepath, custom_objects=dependencies)

test_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_test_sentinel.csv')
history = model.predict_generator(batch_generator(test_df, 1), steps = round(len(test_df)/1))
print(history.shape)
history = history.squeeze()
np.save(pred_file, history)
"""
predictions = np.load(pred_file) #'predictions_unet_sentinel_filled.npy')
y_true = read_imgs_keraspp(test_df).flatten()
y_pred = predictions

get_metrics(y_true, y_pred, binarized=False)

print(predictions.shape)
print(predictions[0].shape)
plt.figure()
for i in range(0, 10):
  prediction = predictions[i]
  print(prediction)
  prediction[prediction > 0.5] = 255
  prediction[prediction != 255] = 0
  print(np.count_nonzero(prediction == 255))
  plt.imshow(prediction)
  plt.axis('off')
  plt.savefig(pred_dir + '/predict_unet' + str(test_df['image'][i].split('.jpeg')[0].split('_')[-1]) + '_pred.png',  bbox_inches = 'tight')
"""
