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
    for i, rows in enumerate(audio):
        if i % frame_size == 0:
            print(i)


def get_audio_from_files(training_amount, parent_dir, sub_dirs):

    file_counter = 0
    audio_files = []
    label_files =  []
    sr = 0
    for fn in glob.iglob('data/cv-valid-train/*.wav'):
        label = ntpath.basename(fn)
        file_counter += 1
        sr, audio = get_audio(fn);
        #audio_files.append(np.array(audio))
        audio_files = np.concatenate([audio_files, audio], axis=0)
        #audio_files = np.concatenate((np.array(audio_files), np.array(audio)))
        label_files.append(1)
        print('Reading: ' + str(file_counter) + ' of ' + str(training_amount))
        if file_counter==training_amount: break;

    frame_generator(sr, audio, 100)
    return sr, np.array(label_files), np.array(audio_files)

def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio

def get_audio_from_model(model, sr, duration, seed_audio):
    print ('Generating audio...')
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1, frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[-1] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print ('Audio generated.')
    return new_audio.astype(np.int16)
