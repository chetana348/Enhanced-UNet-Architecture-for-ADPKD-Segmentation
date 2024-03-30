import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import nibabel as nib
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage import measure
from skimage.transform import resize
from keras_unet.metrics import dice_coef
from keras_unet.models import custom_unet
from keras_unet.losses import jaccard_distance
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps
import fnmatch
import nibabel as nib
import shutil

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=12, dim=(512,512), n_channels=1,
                 n_classes=2, shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temp):
            im_f_name = ID
            lbl_f_name = im_f_name.replace('M.npy', 'K.npy')
            
            im = np.load(im_f_name)
            lbl = np.load(lbl_f_name)

            if self.augment:
                # Randomly flip the image horizontally
                if np.random.rand() > 0.5:
                    im = np.fliplr(im)
                    lbl = np.fliplr(lbl)
                # Randomly flip the image vertically
                if np.random.rand() > 0.5:
                    im = np.flipud(im)
                    lbl = np.flipud(lbl)
                # Randomly rotate the image by 90, 180, or 270 degrees
                num_rotations = np.random.randint(0, 4)
                im = np.rot90(im, k=num_rotations)
                lbl = np.rot90(lbl, k=num_rotations)
                # Randomly shift the image horizontally and vertically by up to 10%
                h_shift = np.random.randint(-int(self.dim[1]*0.1), int(self.dim[1]*0.1))
                v_shift = np.random.randint(-int(self.dim[0]*0.1), int(self.dim[0]*0.1))
                im = np.roll(im, shift=h_shift, axis=0)
                im = np.roll(im, shift=v_shift, axis=1)
                lbl = np.roll(lbl, shift=h_shift, axis=0)
                lbl = np.roll(lbl, shift=v_shift, axis=1)
                # Randomly zoom the image by up to 10%
                #zoom_factor = np.random.uniform(0.9, 1.1)
                #im = resize(im, (int(self.dim[0]*zoom_factor), int(self.dim[1]*zoom_factor)), mode='reflect')
                
                #noise = np.random.normal(loc=0, scale=0.05, size=im.shape)
                #im = im + noise

                X[i, ...,0] = im[...]
                y[i, ...] = lbl[...]
        
        return X, to_categorical(y, num_classes=self.n_classes)
