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

def get_audio_from_files(batch_size, parent_dir, sub_dirs):

    file_counter = 0
    audio_files = []
    sr = 0
    for fn in glob.iglob('data/cv-valid-train/*.wav'):
        label = ntpath.basename(fn)
        file_counter += 1
        sr, audio = get_audio(fn);
        audio_files = np.concatenate([audio_files, audio], axis=0)
        print('Reading: ' + str(file_counter) + ' of ' + str(batch_size))
        if file_counter==batch_size: break;

    # divide audio length by batch size(1 = 1 .wav file)
    frame_size = round(audio_files.shape[0] / batch_size)
    # returns array with audio samples
    frames, label_files = frame_generator(sr, audio_files, frame_size)

    return sr, label_files, frames

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
