"""
Functions for model creations, training and evaluating
Author: Sylle Hoogeveen
"""


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from model import AE, Time_NN, Time_LSTM, Time_GRU


def create_AE(img_inputs, filters, z_num, repeat, num_conv, conv_k, last_k):
    z, out, autoencoder = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)

    autoencoder.compile(loss="mse", optimizer="adam", metrics=["mse"])
    # keras.utils.plot_model(autoencoder, "autoencoder_arch.png")
    return autoencoder

def create_TS_network(x, onum, model):
    if model == 'NN':
        delta_z, TS_network = Time_NN(x, onum)
    if model == 'RNN_LSTM':
        new_z, TS_network = Time_LSTM(x, onum)
    if model== 'RNN_GRU':
        new_z, TS_network = Time_GRU(x, onum)

    TS_network.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    #keras.utils.plot_model(TS_network, "TS_Network_arch.png")
    return TS_network



def train_and_store(model, train_generator,val_generator, batch_size, epochs, callbacks, fig_name):
    # Note: model is stored by using the ModelCheckpoint callback
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

