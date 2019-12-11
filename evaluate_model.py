import sys
from PIL import Image
import scipy
from random import randint
import numpy as np
import pandas as pd
import glob
import math
import warnings
import pdb
#import keras.backend as K
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
#from segmentation_models.metrics import iou_score, dice_score

def dice_coef_burak(y_true, y_pred, smooth=1):
    import keras.backend as k
    y_pred_bool = K.greater(y_pred, 0.5)                                                         
    y_pred = K.cast(y_pred_bool, K.floatx())                                                     
    intersection = K.sum(K.abs(y_true * y_pred))                                                 
    return (intersection / (K.sum(y_true) + K.sum(y_pred) - intersection + 1e-4)) 



def read_imgs_keraspp(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 3))
    imgs_labels = np.zeros((len(imgs_df), 256, 256))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_tensor[index, :, :, :] = np.array(Image.open(row['image']).resize([224, 224]).convert('RGB'))
        imgs_labels[index, :, :] = np.array(Image.open(row['mask']).resize([256, 256]))/255.0

    return imgs_tensor, imgs_labels




def get_metrics(y_true, y_pred, binarized=True):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    assert(y_true.shape == y_pred.shape)
    if not binarized:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred != 1] = 0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    print('Dice/ F1 score:', f1_score(y_true, y_pred, average='binary', pos_label=1))
    dice = np.sum(y_pred[y_true==1])*2.0 / (np.sum(y_true) + np.sum(y_pred))
    print("Dice score:", dice)
#    print('Dice/ F1 score:', dice_coef_burak(y_true, y_pred))
    print('Accuracy score:', accuracy_score(y_true, y_pred))
    print("Precision recall fscore", precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1))

test_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_test_sentinel.csv')
model_name = sys.argv[1]
y_pred = np.load(model_name + '.npy')
X_true, y_true = read_imgs_keraspp(test_df)

#predictions = np.load('predictions_unet_sentinel_filled.npy')
#y_true = read_imgs_keraspp(test_df).flatten()
#y_pred = predictions
y_true = y_true[:-1]
print(y_true[0])
print(y_pred[0])
print(y_pred.shape)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred != 1] = 0
save_file = "outfile" + "_" + model_name + ".jpg"
scipy.misc.imsave('poster_im/outfile_true.jpg',y_true[7])
scipy.misc.imsave('poster_im/' + save_file, y_pred[7])
scipy.misc.imsave('poster_im/outfile_X.jpg', X_true[7])

get_metrics(y_true, y_pred, binarized=True)

