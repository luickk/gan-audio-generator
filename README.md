Generative Adversarial Network Audio generator
===================

The aim is to generate audio, based on the [Common Voice](https://voice.mozilla.org/en/data) dataset using a
[Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network).

----------

#### Résumé

The projects as its whole works quite good, both generator and discriminator are training and competing
against each other. But to achieve acceptable results the generator has to be better than the discriminator, which is not he case. 
Even after 12 Gb of data the discriminator is still way better than the generator which basically means that the generator couldn't 
imitate the sound samples good enough. The expectable result is a monotonous sough. The inability of the generator to get better
than the discriminator can be traced back to the data, a image(black-and-white) imitating GAN for examples works with only one 10 value
per pixel in one image. But one tone(compared to pixel) has 256 16bit values, with a 44 Mhz sample rate, there are a whole of 44000 * 256 * 5 values 
to change in a 5 Sek sound sample, a img generating has only to change 400x400 values, in a 400x400 img to change.

----------

#### Installation

    > - Clone Repository
    > - Install Dependencies
    > - Train

#### Training
    > - Convert .<format> files to .wav files using tools/reformat.py <br>
    > - python main.py -m train

----------

Dependencies
-------------------

> - numpy
> - Keras
> - matplotlib
> - librosa
> - OptionParser
> - uuid
> - tqdm
> - tensorflow
> - scipy
> - sklearn 
> - h5py

Audio Data
-------------------

Samplerate: 44,1 kHz <br>
Audiotype: Mono <br>
Recommended dataset: Common Voice by Mozilla<br>
File format: .wav

Inspired Paper
-------------------

[Continuous recurrent neural networks with adversarial training](https://arxiv.org/pdf/1611.09904.pdf) by <br>
Olof Mogren Chalmers *University of Technology, Sweden*

Dependency specific issues
-------------------

 - librosa

	`raise NoBackendError() ` <br>
    `audioread.NoBackendError` :
    install [ffmpeg](https://ffmpeg.zeranoe.com/builds/) and <br>
    add environment variable for ffmpeg
