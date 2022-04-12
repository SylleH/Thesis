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
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # lots of options to specify
    test_generator = image_gen.flow_from_directory(data_test_dir,
                                                   target_size=(img_height, img_width),
                                                   color_mode="rgb",
                                                   shuffle=False,
                                                   batch_size=10,
                                                   class_mode="input",
                                                   seed=123)
    return test_generator

def create_AE(img_inputs, filters, z_num,
                             repeat, num_conv, conv_k, last_k):
    z, out, autoencoder = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)
    autoencoder.compile(loss="mse", optimizer="adam", metrics=["mse"])
    print(autoencoder.summary())
    # keras.utils.plot_model(autoencoder, "autoencoder_arc.png")
    return autoencoder

def train_and_store(model, train_generator,val_generator, batch_size, epochs, callbacks, fig_name):
    history = model.fit(train_generator, validation_data = val_generator,
                          batch_size = batch_size, epochs= epochs, callbacks = callbacks)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fig_name) #ToDo: find good name lossfunction
    plt.show()
    return history

def predict_and_evaluate(model, test_generator, fig_name):
    #make predictions on test dataset
    predicted = model.predict(test_generator)
    print(str(model))
    #create plot af test data inout and output
    plt.figure(figsize=(20,4))
    img_batch, _ = test_generator.next()
    n  = 10 #number of images to display
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(img_batch[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predicted[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(fig_name) #ToDo: find good name result
    plt.show()


    #retreive test scores
    test_scores = model.evaluate(test_generator)
    return test_scores


def test():
    train_generator, val_generator = create_datagenerators(data_dir, batch_size = 32,img_height = 856,img_width = 784)
    img_batch, _ = train_generator.next()
    first_image = img_batch[0]
    print(first_image.shape)
    plt.figure()
    plt.imshow(first_image)
    plt.show()
