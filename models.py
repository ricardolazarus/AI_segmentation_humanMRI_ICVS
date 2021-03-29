import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping3D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet_2d(input_shape=(256, 256, 1), num_classes=1, kernel_size=3, multiply_feature_maps=2, dropout_rate=0.2):

    inputs = Input(input_shape)

    f, n = kernel_size, multiply_feature_maps
    act = "softmax" if num_classes > 1 else "sigmoid"

    conv1 = Conv2D(32*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(inputs)
    conv11 = Conv2D(32*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
    batch1 = BatchNormalization()(pool1)
    drop1 = Dropout(dropout_rate)(batch1)
    conv2 = Conv2D(64*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop1)
    conv22 = Conv2D(64*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    batch2 = BatchNormalization()(pool2)
    drop2 = Dropout(dropout_rate)(batch2)
    conv3 = Conv2D(128*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop2)
    conv33 = Conv2D(128*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)
    drop3 = Dropout(dropout_rate)(pool3)
    batch3 = BatchNormalization()(drop3)
    conv4 = Conv2D(256*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(batch3)
    conv44 = Conv2D(256*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv44)
    batch4 = BatchNormalization()(pool4)
    conv5 = Conv2D(512*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(batch4)
    conv55 = Conv2D(512*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv5)
    up6 = Conv2D(256*n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv55))
    merge6 = concatenate([conv44, up6], axis=3)
    batch6 = BatchNormalization()(merge6)
    drop6 = Dropout(dropout_rate)(batch6)
    conv6 = Conv2D(256*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop6)
    conv66 = Conv2D(256*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(128*n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv66))
    merge7 = concatenate([conv33, up7], axis=3)
    batch7 = BatchNormalization()(merge7)
    drop7 = Dropout(dropout_rate)(batch7)
    conv7 = Conv2D(128*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop7)
    conv77 = Conv2D(128*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(64*n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv77))
    merge8 = concatenate([conv22, up8], axis=3)
    batch8 = BatchNormalization()(merge8)
    drop8 = Dropout(dropout_rate)(batch8)
    conv8 = Conv2D(64*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop8)
    conv88 = Conv2D(64*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(32*n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv88))
    merge9 = concatenate([conv11, up9], axis=3)
    batch9 = BatchNormalization()(merge9)
    conv9 = Conv2D(32*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(batch9)
    drop9 = Dropout(dropout_rate)(conv9)
    conv99 = Conv2D(32*n, (f, f), activation='relu',padding='same', kernel_initializer='he_normal')(drop9)
    conv10 = Conv2D(num_classes, (1, 1),activation=act)(conv99)
    model = Model(input=inputs, output=conv10)
    return model

def unet_2dslice(input_shape=(256, 256, 1), num_classes=1, kernel_size=3, multiply_feature_maps=2, dropout_rate=0.2):
    f, n = kernel_size, multiply_feature_maps
    model = Sequential()
    model.add(Conv2D(32*n, (f, f), activation='relu', input_shape=(256, 256, 1)))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32* n, (f, f), activation='relu',kernel_initializer='random_uniform'))
    model.add(MaxPooling2D((f, f), strides=(3, 3)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(64*n, (f, f), activation='relu',kernel_initializer='random_uniform'))
    model.add(MaxPooling2D((f, f), strides=(3,3)))
    model.add(Conv2D(96*n, (f, f), activation='relu',kernel_initializer='random_uniform'))
    model.add(MaxPooling2D((f, f), strides=(3, 3)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(128*n, (f, f), activation='relu',kernel_initializer='random_uniform'))
    model.add(MaxPooling2D((f, f), strides=(3, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(256*n, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(256*n, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
        
    return model    