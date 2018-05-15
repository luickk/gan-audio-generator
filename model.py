from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from optparse import OptionParser
from data_proc.data_proc import get_audio_from_files
from sys import getsizeof
from scipy.io.wavfile import read, write
from sklearn.preprocessing import normalize

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os

def build_audio_generator(frame_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=(frame_size, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(256*frame_size))
    model.add(Activation('softmax'))
    model.add(Reshape((frame_size, 256)))

    model.summary()

    noise = Input(shape=(frame_size, 1))

    sound = model(noise)

    return Model(noise, sound)

def build_audio_discriminator(audio_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=(2), padding="same", input_shape=audio_shape))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))

    model.summary()

    audio = Input(shape=audio_shape)

    # Extract feature representation
    features = model(audio)

    # Determine validity and label of the audio
    validity = Dense(1, activation="sigmoid")(features)

    return Model(audio, validity)

def pre_process_datasad(batch_size):
    parent_dir = 'cv-valid-train'
    tr_sub_dirs_training = 'data'
    sr_training, y_train, X_train = get_audio_from_files(batch_size, parent_dir, tr_sub_dirs_training)

    y_train = y_train.reshape(-1, 1)
    return sr_training, y_train, X_train

def pre_process_data(path, batch_size, frame_size, frame_shift):
    parent_dir = 'cv-valid-train'
    tr_sub_dirs_training = 'data'
    sr_training, y_train, X_train = get_audio_from_files(path, batch_size, frame_size, frame_shift, minibatch_size=20)

    return sr_training, y_train, X_train

def train(sr_training, y_train, X_train, generator, discriminator, combined, epochs, batch_size, frame_size):

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        audio = y_train

        half_batch_size = int(audio.shape[1]/2)

        audio_frame = audio[:, audio.shape[1]-frame_size: , :]

        noise = np.random.normal(0, 1, (half_batch, frame_size, 1))

        sampled_labels = 1

        # Generate a half batch of new images
        gen_audio = generator.predict(noise)

        valid = np.ones((1, int(frame_size/2), 1))

        fake = np.zeros((1, int(frame_size/2), 1))

        fake_labels = 10 * np.ones(half_batch).reshape(-1, 1)


        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(audio_frame, valid)
        d_loss_fake = discriminator.train_on_batch(gen_audio, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Sample generator input
        noise = np.random.normal(0, 1, (1, frame_size, 1))

        valid = np.ones((1, int(frame_size/2), 1))

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print (epoch ," Disc loss: " , str(d_loss) , " | Gen loss: " , str(g_loss))

    model_uuid = save_model(generator, discriminator, combined)
    print('Model id: ' + model_uuid)
    new_audio = get_audio_from_model(generator, sr_training, 1, X_train, frame_size)
    write("test.wav", sr_training, new_audio)


def get_audio_from_model(model, sr, duration, seed_audio, frame_size):
    print ('Generating audio...')
    print ('Sample rate: ' + str(sr))
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        pred_audio = model.predict(np.random.normal(0, 1, (1, frame_size, 1)))
        for i in range(pred_audio.shape[1]):
            curr_sample_idx += 1
            if curr_sample_idx > len(new_audio)-1:
                print('Exiting loop')
                break
            pred_audio_sample = pred_audio[0,i,:]

            pred_audio_sample = pred_audio_sample.reshape(256)
            pred_audio_sample /= pred_audio_sample.sum().astype(float)
            predicted_val = np.random.choice(range(256), p=pred_audio_sample)
            ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
            ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15

            new_audio[curr_sample_idx] = ampl_val_16

            pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))

            sys.stdout.write('Percent complete: ' + pc_str + '\r')
            sys.stdout.flush()


    print ('Audio generated.')
    return np.array(new_audio, dtype=np.int16)

def save_model(generator, discriminator, combined):

    model_uuid = str(uuid.uuid1())
    def save(model, model_name):
        model_path = "saved_model/"+model_name+"/model.json"
        weights_path = "saved_model/"+model_name+"/model_weights.hdf5"
        if not os.path.exists(model_path) or not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(model_path))

        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, model_uuid)
    save(discriminator, model_uuid)
    save(combined, model_uuid)

    return model_uuid
