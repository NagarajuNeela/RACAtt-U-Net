import time
import cv2
import scipy.io as sio
import os
import keras
import numpy as np
import random
import tensorflow as tf
from scipy.stats import entropy
from matplotlib import pyplot as plt
from keras import layers
from keras.layers import Activation, Multiply, Input, Conv2D, Conv3D, Dropout, MaxPooling2D, MaxPooling3D, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, Add, Concatenate, Cropping2D, Conv2DTranspose, Flatten, Dense
from keras.models import Model
from sklearn.preprocessing import minmax_scale, MinMaxScaler, StandardScaler
from skimage.util.shape import view_as_windows
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam



# Load .mat file
img = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
gt  = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']


def get_HSI_patches(x, gt, ksize=(32,32), stride=(1, 1), padding='reflect', indix=False):
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (1, n_row, n_clm, n_band))
        y = np.reshape(gt, (1, n_row, n_clm, 1))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        ksizes_ = (1, ksize[0], ksize[1], 1)
        strides_ = (1, stride[0], stride[1], 1)

        x_patches = tf.image.extract_patches(x, ksizes_, strides_, rates=(1, 1, 1, 1), padding='VALID')
        y_patches = tf.image.extract_patches(y, ksizes_, strides_, rates=(1, 1, 1, 1), padding='VALID')

        x_patches = np.reshape(x_patches, (-1, x_patches.shape[-1]))
        x_patches = np.reshape(x_patches, (-1, ksize[0], ksize[1], n_band))

        y_patches = np.reshape(y_patches, (-1, y_patches.shape[-1]))
        y_patches = np.reshape(y_patches, (-1, ksize[0], ksize[1], 1))

        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        y_center_label = np.reshape(y_patches[:, i_1, i_2, :], -1)
        nonzero_index = np.nonzero(y_center_label)
        x_patches_nonzero = x_patches[nonzero_index]
        y_patches_nonzero = y_center_label[nonzero_index]
        if indix is True:
            return x_patches_nonzero, y_patches_nonzero, nonzero_index
        return x_patches_nonzero, y_patches_nonzero

HSI,GT=get_HSI_patches(img, gt)

X_train, X_test, y_train, y_test = train_test_split(HSI, GT)

def conv_block(input_tensor,dilation_rate):
    x = Conv2D(64, (3, 3), dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(64, (3, 3), dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def skipconnection_block(x, g):
    theta_x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
    theta_y = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(theta_x)
    theta_z = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(theta_y)
    theta_z = layers.add([x, theta_z])

    phi_g = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(g)
    phi_h = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(phi_g)
    phi_i = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(phi_h)
    phi_i = layers.add([g, phi_i])

    f1 = Activation('relu')(Add()([theta_z, phi_i]))

    psi_f = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(f1)
    psi_a = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(psi_f)
    psi_b = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(psi_a)
    psi_b = layers.add([f1, psi_b])

    sig_psi_f = Activation('sigmoid')(psi_b)
    sig_psi_f_upsampled = UpSampling2D(size=(1, 1))(sig_psi_f)
    f2 = Activation('relu')(sig_psi_f_upsampled)
    skipconn = Multiply()([f2, x])
    return skipconn


# Define the U-Net architecture
def build_unet(input_shape):
    inputs = Input(shape=(input_shape))
    print("inputs",inputs.shape)

    #Contraction path
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
   

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
   

    conv4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
   

    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
   

    #Expansive path

    up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv5)
   
    crop_height = conv4.shape[1] - up6.shape[1]
    crop_width = conv4.shape[2] - up6.shape[2]

    # Apply Cropping2D layer separately for each dimension
    cropped_conv6 = Cropping2D(cropping=((0, crop_height), (0, crop_width)))(conv4)
    merge6 = skipconnection_block(cropped_conv6, up6)
   
    conv6 = conv_block(merge6, dilation_rate=16)
    conv6 = Dropout(0.2)(conv6)
    conv6 = conv_block(conv6, dilation_rate=16)
    


    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv6)
   
    crop_height = conv3.shape[1] - up7.shape[1]
    crop_width = conv3.shape[2] - up7.shape[2]

    # Apply Cropping2D layer separately for each dimension
    cropped_conv7 = Cropping2D(cropping=((0, crop_height), (0, crop_width)))(conv3)
    merge7 = skipconnection_block(cropped_conv7, up7)
   
    conv7 = conv_block(merge7, dilation_rate=12)
    conv7 = Dropout(0.2)(conv7)
    conv7 = conv_block(conv7, dilation_rate=12)
   

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    crop_height = conv2.shape[1] - up8.shape[1]
    crop_width = conv2.shape[2] - up8.shape[2]

    # Apply Cropping2D layer separately for each dimension
    cropped_conv8 = Cropping2D(cropping=((0, crop_height), (0, crop_width)))(conv2)
    merge8 = skipconnection_block(cropped_conv8, up8)
    
    conv8 = conv_block(merge8, dilation_rate=8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = conv_block(conv8, dilation_rate=8)
   

    flatten_layer = Flatten()(conv8)

    dense_layer1 = Dense(units=64, activation='relu')(flatten_layer)

    dense_layer2 = Dense(units=64, activation='relu')(dense_layer1)

    outputs = Dense(classes, activation='softmax')(dense_layer2)

    model = Model(inputs=inputs, outputs=outputs)
    return model
