import glob
import os
import librosa
import numpy as np
import ntpath
import tqdm
import matplotlib.pyplot as plt

from math import sqrt
from scipy.io.wavfile import read, write
from subprocess import check_output

plt.style.use('ggplot')

def frame_generator(sr, audio, frame_size):
    temp_data = []
    label_files =  []
    for i, rows in enumerate(audio):
        if i % frame_size == 0:
            frame = audio[i-frame_size:i]
            if np.array(frame).shape[0] == frame_size:
                temp_data.append(frame)
                label_files.append(1)
    return np.array(temp_data), np.array(label_files)

def get_audio_from_files(data_path, batch_size, frame_size, frame_shift, minibatch_size=20):

    file_counter = 0
    sr = 0

    for fn in glob.iglob('data/cv-valid-train/*.wav'):

        label = ntpath.basename(fn)
        file_counter += 1
        sr, audio = get_audio(fn);
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

        print('Reading: ' + str(file_counter) + ' of ' + str(batch_size))
        if file_counter==batch_size: break;


    X = np.array(X)
    Y = np.array(Y)

    return sr, X, Y


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio
