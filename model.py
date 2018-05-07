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
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def build_audio_generator(latent_dim, num_classes, audio_shape):
    model = Sequential()
    model.add(LSTM(512, input_dim=latent_dim, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(audio_shape[0]))
    model.add(Activation('tanh'))
    model.add(Reshape((audio_shape[0], 1)))

    model.summary()

    noise = Input(shape=(None, latent_dim,))
    #label = Input(shape=(1,), dtype='int32')
    #label_embedding = Flatten()(Embedding(num_classes, 100)(label))
    #model_input = multiply([noise, label_embedding])

    sound = model(noise)

    return Model([noise], sound)

def build_audio_discriminator(audio_shape, num_classes):
    print(audio_shape)
    model = Sequential()

    model.add(Conv1D(32, kernel_size=(2), padding="same", input_shape=audio_shape))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128))

    model.summary()

    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    audio = Input(shape=audio_shape)

    # Extract feature representation
    features = model(audio)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid")(features)
    #label = Dense(num_classes+1, activation="softmax")(features)

    return Model(audio, [validity])

def pre_process_data(batch_size):
    parent_dir = 'cv-valid-train'
    tr_sub_dirs_training = 'data'
    sr_training, y_train, X_train = get_audio_from_files(batch_size, parent_dir, tr_sub_dirs_training)

    y_train = y_train.reshape(-1, 1)
    return sr_training, y_train, X_train

def train(sr_training, y_train, X_train, generator, discriminator, combined, epochs, batch_size):

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        audio = X_train[idx]
        imgs = X_train[idx]
        #noise = Input(shape=(None, latent_dim,))
        noise = np.random.normal(0, 1, (1, half_batch, 100))

        # The labels of the digits that the generator tries to create an
        # image representation of
        sampled_labels = 1

        # Generate a half batch of new images
        gen_imgs = generator.predict([noise])

        valid = np.ones((half_batch, 128596, 1))
        fake = np.zeros((half_batch, 128596, 1))

        # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
        img_labels = y_train[idx]
        fake_labels = 10 * np.ones(half_batch).reshape(-1, 1)


        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Sample generator input
        #noise = np.random.normal(0, 1, (batch_size, 100))
        noise = np.random.normal(0, 1, (1, batch_size, 100))

        #valid = np.ones((batch_size, 1))
        valid = np.ones((1, 128596, 1))
        # Generator wants discriminator to label the generated images as the intended
        # digits
        #sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

        print(noise.shape)
        print(valid.shape)
        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print (epoch ," Disc loss: " , str(d_loss) , " | Gen loss: " , str(g_loss))

    save_model(generator, discriminator, combined)
    #sample_images(generator, epoch)



def sample_images(generator, epoch):
    r, c = 10, 10
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.array([num for _ in range(r) for num in range(c)])

    gen_imgs = generator.predict([noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    print('Creating imgs')
    plt.close()

def save_model(generator, discriminator, combined):

    def save(model, model_name):
        model_path = "saved_model/"+model_name+".json"
        weights_path = "saved_model/"+model_name+"_weights.hdf5"
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "mnist_acgan_generator")
    save(discriminator, "mnist_acgan_discriminator")
    save(combined, "mnist_acgan_adversarial")
