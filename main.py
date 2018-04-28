from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from model import build_discriminator, build_generator, train, save_model
from optparse import OptionParser

import matplotlib.pyplot as plt

import numpy as np

def main():

    parser = OptionParser()

    # Only required for labeling - Defines train or generate mode
    parser.add_option('-m', '--mode', help='train or gen', dest='mode', default = 'label')
    # Only required for labeling - Enter Model id here
    parser.add_option('-u', '--uid', help='enter model id here')

    (options, args) = parser.parse_args()

    if options.mode == 'train':
        # Input shape
        img_rows = 28
        img_cols = 28
        channels = 1
        img_shape = (img_rows, img_cols, channels)
        num_classes = 10
        latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        discriminator = build_discriminator(img_shape, num_classes)
        discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        generator = build_generator(latent_dim, channels, num_classes)

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
        combined.compile(loss=losses,
            optimizer=optimizer)
        #14000
        train(generator, discriminator, combined, epochs=5000, batch_size=32, sample_interval=200)


if __name__ == '__main__':
    main()
