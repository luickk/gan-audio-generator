from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from model import build_discriminator, build_generator, train, save_model, pre_process_data, build_audio_generator, build_audio_discriminator
from optparse import OptionParser
import uuid
from tqdm import tqdm
import librosa
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

def main():

    parser = OptionParser()

    # Only required for labeling - Defines train or generate mode
    parser.add_option('-m', '--mode', help='train or gen', dest='mode', default = 'label')
    # Only required for labeling - Enter Model id here
    parser.add_option('-u', '--uid', help='enter model id here')

    batch_size = 2

    epochs = 100

    (options, args) = parser.parse_args()

    if options.mode == 'train':
        sr_training, y_train, X_train_raw = pre_process_data(batch_size)

        num_classes = 1

        latent_dim = 100

        # reshaping array
        X_train = X_train_raw.reshape(X_train_raw.shape[0],X_train_raw.shape[1], 1)
        audio_shape = (X_train.shape[1],1)

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy']

        # Build and compile the discriminator
        audio_discriminator = build_audio_discriminator(audio_shape, num_classes)
        audio_discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        audio_generator = build_audio_generator(latent_dim, num_classes, audio_shape)

        # The generator takes noise and the target label as input
        noise = Input(shape=(None, latent_dim,))

        audio = audio_generator([noise])

        # For the combined model we will only train the generator
        audio_discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image

        audio_valid = audio_discriminator(audio)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates audio => determines validity
        audio_combined = Model([noise], [audio_valid])
        audio_combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        train(sr_training, y_train, X_train, audio_generator, audio_discriminator, audio_combined, epochs, batch_size)


if __name__ == '__main__':
    main()
