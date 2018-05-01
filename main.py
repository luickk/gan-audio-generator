from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from model import build_discriminator, build_generator, train, save_model, pre_process_data, build_audio_generator
from optparse import OptionParser
import uuid
from tqdm import tqdm
import librosa

import matplotlib.pyplot as plt

import numpy as np

def main():

    parser = OptionParser()

    # Only required for labeling - Defines train or generate mode
    parser.add_option('-m', '--mode', help='train or gen', dest='mode', default = 'label')
    # Only required for labeling - Enter Model id here
    parser.add_option('-u', '--uid', help='enter model id here')

    batch_size = 32

    epochs = 100

    (options, args) = parser.parse_args()

    if options.mode == 'train':
        sr_training, y_train, X_train_raw = pre_process_data(batch_size)

        num_classes = 1

        # Input shape
        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)
        #num_classes = 10
        latent_dim = 100

        # reshaping array
        X_train = X_train_raw.reshape(1,X_train_raw.shape[0],X_train_raw.shape[1])

        print(X_train.shape)

        audio_shape = X_train.shape


        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']


        # Build and compile the discriminator
        discriminator = build_discriminator(img_shape, num_classes)
        discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])


        # Build the generator
        generator = build_generator(latent_dim, channels, num_classes)
        audio_generator = build_audio_generator(audio_shape, num_classes, latent_dim)

        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(latent_dim,))
        label = Input(shape=(1,))
        img = generator([noise, label])

        # For the combined model we will only train the generator
        discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        combined = Model([noise, label], [valid, target_label])
        combined.compile(loss=losses, optimizer=optimizer)

        #14000
        train(sr_training, y_train, X_train, generator, discriminator, combined, epochs, batch_size)


if __name__ == '__main__':
    main()
