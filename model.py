
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from optparse import OptionParser
from data_proc.data_proc import get_audio_from_files, get_audio
from sys import getsizeof
from scipy.io.wavfile import read, write
from sklearn.preprocessing import normalize

import sys, os
import glob
import os
import sys
import tensorflow as tf
import numpy as np
import uuid
import ntpath
import matplotlib.pyplot as pyplot

def build_audio_generator(frame_size):
    model = Sequential()
    model.add(LSTM(256, input_shape=(frame_size, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
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

def train(generator, discriminator, combined, epochs, frame_size, frame_shift):


    file_counter = 0
    epoch_counter = 0
    sr = 0
    sr_training = 0
    g_metrics = []
    d_metrics = []

    for fn in glob.iglob('data/cv-valid-train/*.wav'):
        label = ntpath.basename(fn)
        file_counter += 1
        sr, audio = get_audio(fn);
        sr_training = sr
        audio_len = len(audio)

        X_temp = []
        Y_temp = []

        X = []
        Y = []

        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            if len(temp.shape) > 0:
                print('Mono audio data as input required, no stereo!')
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (np.log(1+256))) + 1)/2.0 * 255)

            X_temp.append(frame.reshape(frame_size, 1))
            Y_temp.append((np.eye(256)[target_val]))

            X_temps = np.array(X_temp)
            Y_temps = np.array(Y_temp)

        X.append(X_temp)
        Y.append(Y_temp)

        Y = np.array(Y)

        if(Y.shape[1] > frame_size):

            if epoch_counter>epochs: break;
            for s in range(0, Y.shape[1], frame_size):

                audio = Y
                epoch_counter += 1


                if epoch_counter>epochs: break;

                audio_frame = audio[:, s:s+500, :]
                if audio_frame.shape[1]<frame_size: break;

                noise = np.random.normal(0, 1, (1, frame_size, 1))


                # Generate a half batch of new images
                gen_audio = generator.predict(noise)

                valid = np.ones((1, int(frame_size/2), 1))

                fake = np.zeros((1, int(frame_size/2), 1))


                # Train the discriminator
                d_acc_real = discriminator.train_on_batch(gen_audio, valid)
                d_acc_fake = discriminator.train_on_batch(audio_frame, fake)
                d_loss = 0.5 * np.add(d_acc_real, d_acc_fake)

                d_metrics.append(d_loss)

                # Train the generator
                g_loss = combined.train_on_batch(noise, valid)
                g_metrics.append(g_loss)

                # Plot the progress
                print(str(epoch_counter) + '/' + str(epochs) + ' > Discriminator loss: ' + str(d_loss) + ' | Generator loss: ' +  str(g_loss))
            else:
                print('Not enough data frames per file, decrease frame_size!')
                break

    model_uuid = save_model(generator, discriminator, combined, g_metrics, d_metrics)
    print('Model id: ' + model_uuid)
    new_audio = get_audio_from_model(generator, sr_training, 1, frame_size)
    write('saved_model/'+model_uuid+'/output.wav', sr_training, new_audio)


def get_audio_from_model(model, sr, duration, frame_size):
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

def save_model(generator, discriminator, combined, g_metrics, d_metrics):
    model_uuid = str(uuid.uuid1())
    model_path = ""
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
    if os.environ.get('DISPLAY','') == '':
        print('no display found')
    else:	
	pyplot.plot(g_metrics, label="G loss")
    	pyplot.plot(d_metrics, label="D loss")
    	pyplot.legend(loc='upper left')
    	pyplot.savefig('saved_model/' + model_uuid + '/graph.png')

    return model_uuid
