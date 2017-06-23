from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:45:01 2017

@author: kzhang
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import argparse
import os
import numpy as np
from PIL import Image
import math
import pickle
import cv2


from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import SGD


# define training params
batch_size = 64
num_epochs = 100
image_size = [64, 64, 1]
noise_vector_dim = 100
mode = 'train'
resume = False
files = os.listdir('pic64')

# defines how many batches later to save output
save_interval = 50

# define output params
output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[0, :, :]
    return image

generator = Sequential()
generator.add(Dense(input_dim=noise_vector_dim, output_dim=1024))
generator.add(Activation('tanh'))
generator.add(Dense(128 * 16 * 16))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))
generator.add(Reshape((128, 16, 16), input_shape=(128 * 16 * 16,)))
generator.add(UpSampling2D(size=(2, 2), dim_ordering="th"))
generator.add(Convolution2D(64, 5, 5, border_mode='same', dim_ordering="th"))
generator.add(Activation('tanh'))
generator.add(UpSampling2D(size=(2, 2), dim_ordering="th"))
generator.add(Convolution2D(1, 5, 5, border_mode='same', dim_ordering="th"))
generator.add(Activation('tanh'))

# define discriminator model
discriminator = Sequential()
discriminator.add(Convolution2D(64, 5, 5,
                                border_mode='same',
                                input_shape=(1, 64, 64),
                                dim_ordering="th"))
discriminator.add(Activation('tanh'))
discriminator.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
discriminator.add(Convolution2D(128, 5, 5, border_mode='same', dim_ordering="th"))
discriminator.add(Activation('tanh'))
discriminator.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
discriminator.add(Flatten())
discriminator.add(Dense(1024))
discriminator.add(Activation('tanh'))
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))

# define gan model by connecting generator output to discriminator input
gan = Sequential()
gan.add(generator)
discriminator.trainable = False
gan.add(discriminator)

training_history = {'discriminator': [],
                    'gan': []}

if mode == 'train':
    # load mnist data and convert to float32 with range -1 to 1
    i = 0
    X_train = np.zeros(shape=(len(files), 64, 64))
    for file in files:
        X_train[i] = cv2.imread('pic64/' + file, 0)
        i += 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    # define optimizers
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # compile the models
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    gan.compile(loss='binary_crossentropy', optimizer=g_optim)
    # once gan is compiled,
    # set discriminator back to trainable (we alternate between training generator and discriminator)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    if resume:
        # load pretrained weights to continue from
        generator.load_weights('generator_final.h5')
        discriminator.load_weights('discriminator_final.h5')

    noise = np.zeros((batch_size, noise_vector_dim))

    num_batches = int(X_train.shape[0] / batch_size)
    # perform training
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        for index in range(num_batches):
            noise = np.random.uniform(-1, 1, (batch_size, noise_vector_dim))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose=0)
            if (index + 1) % save_interval == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    os.path.join(output_dir, "image_{}_{}.png".format(epoch + 1, index + 1)))
            X = np.concatenate((image_batch, generated_images))
            y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))

            d_loss = discriminator.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, (batch_size, noise_vector_dim))
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(noise, np.ones(batch_size))
            discriminator.trainable = True

            print(
                "Trained batch {}/{}: Discriminator loss = {}, GAN loss = {}".format(index + 1, num_batches, d_loss,
                                                                                     gan_loss))
            training_history['discriminator'].append(d_loss)
            training_history['gan'].append(gan_loss)
        # generator.save_weights(os.path.join(output_dir, 'generator_{}.h5'.format(epoch + 1)),
        #                        True)
        # discriminator.save_weights(
        #     os.path.join(output_dir, 'discriminator_{}.h5'.format(epoch + 1)), True)
        # with open(os.path.join(output_dir, 'partial_training_history_{}.pickle'.format(epoch + 1)), 'wb') as f:
        #     pickle.dump(training_history, f)
        # print("Generator & Discriminator weights, & Training history so far saved in directory = {}".format(
        #     output_dir))
    generator.save_weights(os.path.join(output_dir, 'generator_final.h5'), True)
    discriminator.save_weights(os.path.join(output_dir, 'discriminator_final.h5'), True)
    with open(os.path.join(output_dir, 'training_history.pickle'), 'wb') as f:
        pickle.dump(training_history, f)

elif mode == 'generate':
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights(os.path.join(output_dir, 'generator_final.h5'))
    if pretty:
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights(os.path.join(output_dir, 'discriminator_final.h5'))

        # plot_model(generator, to_file='generator_model.png')
        # plot_model(discriminator, to_file='discriminator_model.png')
        noise = np.zeros((batch_size * 20, noise_vector_dim))
        for i in range(batch_size * 20):
            noise[i, :] = np.random.uniform(-1, 1, noise_vector_dim)
        generated_images = generator.predict(noise, verbose=1)
        d_pred = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pred, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        pretty_images = np.zeros((batch_size, 1) +
                                 (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(batch_size)):
            idx = int(pre_with_index[i][1])
            pretty_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(pretty_images)
    else:
        noise = np.zeros((batch_size, noise_vector_dim))
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, noise_vector_dim)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(output_dir, "generated_image.png"))

else:
    print("INVALID MODE SPECIFIED!!")