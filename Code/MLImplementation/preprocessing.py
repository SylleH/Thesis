"""
Preprocessing functions for CFD simulations to tensor input for model
Author: Sylle Hoogeveen
"""

from ops import *
import tensorflow as tf
import matplotlib.pyplot as plt
from model import AE


# scenario = "straight" #bifurcation, bend90 or branch
# data_dir = "data/"+scenario+"/"

def create_datagenerators(data_dir, batch_size, img_height, img_width):
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

def create_AE(img_inputs, filters, z_num,
                             repeat, num_conv, conv_k, last_k):
    z, out, autoencoder = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)
    autoencoder.compile(loss="mse", optimizer="adam", metrics=["mse"])
    print(autoencoder.summary())
    # keras.utils.plot_model(autoencoder, "autoencoder_arc.png")
    return autoencoder



def test():
    train_generator, val_generator = create_datagenerators(data_dir, batch_size = 32,img_height = 856,img_width = 784)
    img_batch, _ = train_generator.next()
    first_image = img_batch[0]
    print(first_image.shape)
    plt.figure()
    plt.imshow(first_image)
    plt.show()

print(tf.config.list_physical_devices())