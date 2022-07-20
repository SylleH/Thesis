"""
Functions for model creations, loading, training and evaluating
Author: Sylle Hoogeveen
"""


import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
from model import AE, Time_NN, Time_LSTM, Time_GRU
import os
from datetime import date
import numpy as np


def create_AE(img_inputs, filters, z_num, repeat, num_conv, conv_k, last_k, hp=None):
    """
    Function creates Autoencoder model als defined in model.py file
    Also allows for hyperparameter tuning if hp is defined
    """
    #img_inputs = keras.Input(shape=(192, 256, 1))  # shape of each sample needs to be supplied, this is after cropping
    # Autoencoder with fixed parameters
    z, out, autoencoder, encoder_model, generator_model = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)

        # # Hyperparameters for tuning
        # hp_filters = hp.Int('nodes', min_value=8, max_value=32, step=8)
        # hp_BB = hp.Int('BigBlocks', min_value=2, max_value=6, step=2)
        # hp_SB = hp.Int('SmallBlocks', min_value=2, max_value=5, steps=1)
        # hp_zdim = hp.Int('z_num', min_value = 25, max_value=150, steps=25)
        # hp_kernel = hp.Int('conv_k', min_value = 2, max_value = 4, steps =1)
        #
        # z, out, autoencoder, encoder_model, generator_model = AE(img_inputs, filters=hp_filters, z_num=hp_zdim,
        #                      repeat=hp_BB, num_conv=hp_SB, conv_k=hp_kernel, last_k=last_k)

    autoencoder.compile(loss="mse", optimizer="adam", metrics=["mse"])
    # keras.utils.plot_model(autoencoder, "autoencoder_arch.png")
    return autoencoder

def create_TS_network(x, onum, numlayers, nodenum, dropout, model):
    """
    Function creates Timeseries network, either Neural Network, RNN LSTM network or RNN GRU network
    as defined in model.py file
    Also allows for hyperparamter tuning if hp is defined
    """
    if model == 'NN':
        delta_z, TS_network = Time_NN(x, onum, numlayers, nodenum, dropout)
    if model == 'RNN_LSTM':
        new_z, TS_network = Time_LSTM(x, onum)
    if model== 'RNN_GRU':
        new_z, TS_network = Time_GRU(x, onum)

    TS_network.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    #keras.utils.plot_model(TS_network, "TS_Network_arch.png")
    return TS_network


def train_and_store(model, train_generator,val_generator, batch_size, epochs, callbacks,fig_name):
    # Note: model is stored by using the ModelCheckpoint callback
    history = model.fit(train_generator, validation_data = val_generator,
                          batch_size = batch_size, epochs= epochs,  callbacks = callbacks)

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


def predict_and_evaluate(model, test_set, fig_name, pipe, images=None, encoded=None, num=None ):
    #matplotlib.use('agg')

    #make predictions on test dataset
    predicted = model.predict(test_set)
    print(predicted.shape)
    predicted_beat= np.empty([20,192,256,1])
    for i in range(20):
        predicted = model.predict(predicted)
        print(predicted[0].shape)
        predicted_beat[i,:,:,:] = predicted[0]
        print(predicted_beat.shape)



    #if no images are supplied, the autoencoder is given as model (eval - AE pipeline)
    if pipe == 'AE':
        img_batch, _ = test_set.next()
        test_scores = model.evaluate(test_set)
        plt.figure(figsize=(20, 4))
        plt.suptitle('test_loss =' + str(test_scores[0]))
        r = 2
    #if images are supplied, the decoder is given as model (eval - total_sep pipeline), encoded is supplied
    #or total network is given as model (eval - total_comb pipeline), encoded is not supplied
    elif pipe == 'total_sep':
        img_original_batch, _ = images.next()
        img_decoded = model.predict(encoded)
        test_scores = None
        plt.figure(figsize=(20,6))
        #plt.suptitle("Feature " + str(num))
        r = 3
    elif pipe == 'total_com':
        img_batch = test_set
        test_scores = model.evaluate(x=test_set, y=images)
        plt.figure(figsize=(20, 6))
        plt.suptitle('test_loss =' + str(test_scores[0]))
        r = 3

    n  = 10 #number of images to display
    for i in range(n):
        # Display original
        ax = plt.subplot(r, n, i+1)

        # Need to shift the index for comparison
        if pipe == 'total_sep':
            plt.imshow(img_original_batch[i])
        else:
            plt.imshow(img_batch[i])
        plt.gray()
        #plt.title("timestep: "+str(i*5))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction or prediction of TS network
        ax = plt.subplot(r, n, i + 1 + n)
        plt.imshow(predicted_beat[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction if previous is TS network prediction
        if pipe == 'total_sep':
            ax = plt.subplot(r ,n, i+1+2*n)
            plt.imshow(img_decoded[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if pipe == 'total_com':
            ax = plt.subplot(r ,n, i+1+2*n)
            plt.imshow(predicted_beat[i+10])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


    plt.savefig(fig_name) #ToDo: find good name result
    plt.show()
    return test_scores

def load_model(network, checkpoint_filepath, date=str(date.today())):
    checkpoint_filepath = 'model/' + checkpoint_filepath
    print(network)
    print(checkpoint_filepath)
    print(date)

    if os.path.exists(os.path.join(checkpoint_filepath, 'model_' + date)):
        model = keras.models.load_model(os.path.join(checkpoint_filepath, 'model_' + date))
        print(model.summary())

        if network == "AE":
            encoder = keras.Model(model.input, model.get_layer(name='encoded').output)
            encoder.compile(optimizer="Adam", loss="mse", metrics=["mae"])
            decoder = keras.Model(model.get_layer(name='encoded').output, model.get_layer(name='decoded').output)
            decoder.compile(optimizer="Adam", loss="mse", metrics=["mae"])
            return model, encoder, decoder

        elif network == "TS":
            return model

    else:
        print('train model first!')