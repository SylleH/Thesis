"""
Functions for model creations, training and evaluating
Author: Sylle Hoogeveen
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from model import AE
from ops import custom_save
import os

def create_AE(img_inputs, filters, z_num, repeat, num_conv, conv_k, last_k):
    z, out, autoencoder = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)
    setattr(autoencoder, 'save', custom_save)

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
    plt.savefig(fig_name)
    plt.show()
    return history

def predict_and_evaluate(model, test_generator, fig_name):
    #make predictions on test dataset
    predicted = model.predict(test_generator)

    #retreive test scores
    test_scores = model.evaluate(test_generator)

    #create plot af test data inout and output
    plt.figure(figsize=(20,4))
    plt.suptitle('val_loss =' + str(test_scores[0]))
    img_batch, _ = test_generator.next()
    n  = 1 #number of images to display
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(img_batch[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predicted[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(fig_name) #ToDo: find good name result
    plt.show()



    return test_scores
