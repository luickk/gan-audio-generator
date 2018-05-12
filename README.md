Generative Adversarial Network Audio generator
===================

The aim is to generate audio based on the [Common Voice](https://voice.mozilla.org/en/data) dataset using a
[Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network).

----------

#### <i class="icon-down-big"></i> Installation

	> - Clone Repository
	> - Install Dependencies

#### <i class="icon-ccw"></i> Training

	> - python main.py -m train

#### <i class="icon-right-big"></i> Testing

	> - python main.py -u <model ID from trained model here> -i <file name> -m label


----------

Dependencies
-------------------

> - numpy
> - Keras
> - matplotlib
> - librosa


Audio Data
-------------------

Samplerate: 44 kHz <br>
Audiotype: Mono <br>
Recommended dataset: Common Voice by Mozilla

Dependency specific issues
-------------------

 - librosa

	`raise NoBackendError() ` <br>
    `audioread.NoBackendError` :
    install [ffmpeg](https://ffmpeg.zeranoe.com/builds/) and <br>
    add environment variable for ffmpeg

Example
-------------------
// TODO
