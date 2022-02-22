"""
Preprocessing functions for CFD simulations to tensor input for model
Author: Sylle Hoogeveen
"""

import tensorflow as tf
# import tensorflow_datasets as tfds
# import numpy as np
from ops import *
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

scenario = "straight" #bifurcation, bend90 or branch
data_dir = "data/"+scenario+"/cropped/"



def create_cropped_images():
    for filename in glob.glob("*.png"):
        file, ext = os.path.splitext(filename)
        im = Image.open(filename)
        cropped_im = im.crop((397,48,1181,904))
        cropped_im.save(data_dir+file+"_cropped.png")


def create_datasets(data_dir, batch_size, img_height, img_width):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2) #lots of options to specify
    train_generator = image_gen.flow_from_directory(data_dir,
                                                    target_size=(img_height,img_width),
                                                    color_mode="rgb",
                                                    shuffle=True,
                                                    subset="training",
                                                    batch_size = batch_size,
                                                    class_mode="input",
                                                    seed = 123)

    val_generator = image_gen.flow_from_directory(data_dir,
                                                    target_size=(img_height,img_width),
                                                    color_mode="rgb",
                                                    shuffle=True,
                                                    subset="validation",
                                                    batch_size=batch_size,
                                                    class_mode="input",
                                                    seed = 123)

    return train_generator, val_generator

def crop(image):
    print(image.shape)
    start_y = 48
    start_x = 397
    cropped_image = image[start_y:(952 - start_y),start_x:(1578 - start_x), :]
    print(cropped_image.shape)
    return cropped_image

def preprocess(ds):
    #normalization_layer = tf.keras.layers.Rescaling(1./255)
    crop_layer = tf.keras.layers.Cropping2D(cropping=(48,397)) #(symmetric_height_crop, symmertric_width_crop)

    #ds = ds.map(lambda x: (normalization_layer(x)))
    preprocessed_ds = ds.map(lambda x: (crop_layer(x)))

    return preprocessed_ds


def test():
    train_ds, _ = create_datasets(data_dir, batch_size = 32,img_height = 952,img_width = 1578)
    img_batch = next(iter(train_ds))
    first_image = img_batch[0]
    print(first_image.shape)
    plt.figure()
    plt.imshow(first_image.numpy().astype("uint8"))
    plt.show()

    preprocessed_ds = preprocess(train_ds)
    img_batch = next(iter(preprocessed_ds))
    first_image = img_batch[0]
    plt.figure()
    plt.imshow(first_image.numpy().astype("uint8"))
    plt.show()
