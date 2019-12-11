import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        train_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_train_sentinel.csv')
        val_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_val_sentinel.csv')
        df = val_df
        if data_type == "train":
            df = train_df
      
        batch_images = df.sample(n=batch_size) # d$np.random.choice(path, size=batch_size)
        print(batch_images)
        imgs_A = []
        imgs_B = []
        for index, img_path_tpl in batch_images.iterrows():
            img_B, img_A = self.imread(img_path_tpl['image']), self.imread(img_path_tpl['mask'])

#            h, w, _ = img.shape
#            _w = int(w/2)
#            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/255. - 1.
        imgs_B = np.array(imgs_B)/255. - 1.

        return imgs_A, imgs_B


    def load_batch_generator(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        train_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_train_sentinel.csv')
        val_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_test_sentinel.csv')
        #print(val_df)
        df = val_df
        if data_type == "train":
            df = train_df
        self.n_batches = int(len(df.index) / batch_size)

        for i in range(self.n_batches-1):
            batch = df.iloc[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for index, row in batch.iterrows():
#            for img in batch:
                img_B = self.imread(row['image'])
                img_A = self.imread(row['mask'])
#                h, w, _ = img.shape
#                half_w = int(w/2)
#                img_A = img[:, :half_w, :]
#                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/255. - 1.
            imgs_B = np.array(imgs_B)/255. - 1.

            scipy.misc.imsave('outfile_true_B.jpg', imgs_B[0])
            yield imgs_B, imgs_A

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        train_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_train_sentinel.csv')
        val_df = pd.read_csv('./data/sentinel/sentinel_old/parcel_segmentation_test_sentinel.csv')
        #print(val_df)
        df = val_df
        if data_type == "train":
            df = train_df
        self.n_batches = int(len(df.index) / batch_size)

        for i in range(self.n_batches-1):
            batch = df.iloc[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for index, row in batch.iterrows():
#            for img in batch:
                img_B = self.imread(row['image'])
                img_A = self.imread(row['mask'])
#                h, w, _ = img.shape
#                half_w = int(w/2)
#                img_A = img[:, :half_w, :]
#                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/255. - 1.
            imgs_B = np.array(imgs_B)/255. - 1.

            scipy.misc.imsave('outfile_true_B.jpg', imgs_B[0])
            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
