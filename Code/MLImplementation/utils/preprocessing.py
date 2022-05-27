"""
Preprocessing functions for CFD simulations images to tensor input for model
Author: Sylle Hoogeveen
"""

from ops import *
import tensorflow as tf
import matplotlib.pyplot as plt


# scenario = "straight" #bifurcation, bend90 or branch
# data_dir = "data/"+scenario+"/"

def create_train_val_datagen(data_dir, batch_size, img_height, img_width):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2) #lots of options to specify
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

def create_test_datagen(data_test_dir, img_height, img_width):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # lots of options to specify
    test_generator = image_gen.flow_from_directory(data_test_dir,
                                                   target_size=(img_height, img_width),
                                                   color_mode="rgb",
                                                   shuffle=False,
                                                   batch_size=10,
                                                   class_mode="input",
                                                   seed=123)
    return test_generator



def test():
    train_generator, val_generator = create_train_val_datagen(data_dir, batch_size = 32,img_height = 856,img_width = 784)
    img_batch, _ = train_generator.next()
    first_image = img_batch[0]
    print(first_image.shape)
    plt.figure()
    plt.imshow(first_image)
    plt.show()
