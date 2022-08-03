"""
Functions for model creations, loading, training and evaluating
Author: Sylle Hoogeveen
"""


import tensorflow as tf
from tensorflow import keras
from utils.ops import *
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt
from model import AE, Time_NN, Time_LSTM, Time_GRU
import os
from datetime import date
import numpy as np
import cv2


def create_AE(img_inputs, filters, z_num, repeat, num_conv, conv_k, last_k, hp=None):
    """
    Function creates Autoencoder model als defined in model.py file
    Also allows for hyperparameter tuning if hp is defined
    """
    #img_inputs = keras.Input(shape=(192, 256, 1))  # shape of each sample needs to be supplied, this is after cropping
    # Autoencoder with fixed parameters
    z, out, autoencoder, encoder_model, generator_model = AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)



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
    predicted, predicted2, predicted3, predicted4, predicted5 = model.predict(test_set)
    #print(predicted.shape)
    # predicted_beat= np.empty([20,192,256,1])
    # for i in range(20):
    #     predicted = model.predict(predicted)
    #     print(predicted[0].shape)
    #     predicted_beat[i,:,:,:] = predicted[0]
    #     print(predicted_beat.shape)



    #if no images are supplied, the autoencoder is given as model (eval - AE pipeline)
    if pipe == 'AE':
        img_batch, _ = test_set.next()
        test_scores = model.evaluate(test_set)
        plt.figure(figsize=(20, 4))
        plt.suptitle('test_loss =' + str(test_scores[0]))
        n = 10  # number of images to display
        r = 2
    #if images are supplied, the decoder is given as model (eval - total_sep pipeline), encoded is supplied
    #or total network is given as model (eval - total_comb pipeline), encoded is not supplied
    elif pipe == 'total_sep':
        img_original_batch, _ = images.next()
        img_decoded = model.predict(encoded)
        test_scores = None
        plt.figure(figsize=(20,6))
        #plt.suptitle("Feature " + str(num))
        n = 10  # number of images to display
        r = 3
    elif pipe == 'total_com':
        img_batch, _ = test_set[0]
        test_scores = model.evaluate(test_set)
        plt.figure(figsize=(20, 4))
        plt.suptitle('test_loss =' + str(test_scores[0]))
        n = 10  # number of images to display
        r = 2

    for i in range(n):
        # Display original
        if not pipe == 'total_com':
            ax = plt.subplot(r, n, i+1)
        else:
            ax = plt.subplot(r, n, 1)

        # Need to shift the index for comparison
        if pipe == 'total_sep':
            plt.imshow(img_original_batch[i])
        elif pipe == 'total_com':
            plt.imshow(img_batch[10])
        else:
            plt.imshow(img_batch[i])
        plt.gray()
        #plt.title("timestep: "+str(i*5))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction or prediction of TS network
        ax = plt.subplot(r, n, i + 1 + n)
        plt.imshow(predicted[i])

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
        # if pipe == 'total_com':
        #     ax = plt.subplot(r ,n, i+1+2*n)
        #     plt.imshow(predicted_beat[i+10])
        #     plt.gray()
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)


    plt.savefig(fig_name) #ToDo: find good name result
    plt.show()
    return test_scores

def predict_and_plot_total(model, test_set, predicted_ts):
    predicted1, predicted2, predicted3, predicted4, predicted5 = model.predict(test_set)
    predicted_list = [predicted1, predicted2, predicted3, predicted4, predicted5]

    img_batch,_ = test_set[0]
    test_scores = model.evaluate(test_set)
    print('total_test_loss =' + str(test_scores[0]))
    n = int(predicted_ts*2)  # number of images to display
    r = 3
    for i in range(int(len(img_batch)/6)):
        plt.figure(figsize=(20, 6))
        plt.suptitle(f"predictions {(i*2)*6} and {(2*i+1)*6}")
        ax = plt.subplot(r, n, 1)
        ax.title.set_text(f"t={(i*2)*6}")
        plt.imshow(img_batch[(i*2)*6])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(r, n, 6)
        ax.title.set_text(f"t={(2*i+1)*6}")
        plt.imshow(img_batch[(2*i+1)*6])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j in range(predicted_ts):
            ax = plt.subplot(r, n, j+1+n)
            ax.title.set_text(f"t={(i * 12)+1+j}")
            plt.imshow(predicted_list[j][(i*12)])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        for j in range(predicted_ts):
            ax = plt.subplot(r,n,j+predicted_ts+1+n)
            ax.title.set_text(f"t={((2*i+1)*6) + 1 + j}")
            plt.imshow(predicted_list[j][(2*i+1)*6])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()



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

class ErrorMetrics():

    def __init__(self, images, timeseries, prediction):
        self.images,_ = images[0]
        self.image = self.images[0]
        self.timeseries = timeseries
        self.prediction = prediction[0]

        gray = cm.get_cmap('gray')
        self.colormap_lookup = gray(range(256))[:,0]

    def boundary_check(self):
        true_boundary, _ = find_boundary(self.image)
        pred_boundary, _ = find_boundary(self.prediction)
        #ToDo: find measure for boundary displacement? Array length contour is unequal
        if not np.array_equal(true_boundary, pred_boundary):
            boundary_error = 1
        else:
            boundary_error = 0
        return boundary_error

    def noslip(self):
        # image = cv2.imread('../data/NN_testset/straight/straight_w11.0_v3.5_2000.png', cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(src=image, dsize=(256,192))
        side, _, _ = find_walls_inlet_outlet(self.prediction)

        #ToDo: discuss if normalization is legal or cheating, otherwise cut off...?
        prediction = (self.prediction - np.min(self.prediction)) / np.ptp(self.prediction)
        prediction = prediction * 255
        prediction = prediction.astype(np.uint8)

        side_values = [prediction[coord] for coord in side]
        side_values.pop(0)
        side_values.pop(-1)
        #convert from pixel color error to velocity error
        side_values = [self.colormap_lookup[value[0]]*0.5 for value in side_values]

        # print(side)
        print(side_values)
        slip_violation = sum(side_values)
        print(slip_violation)
        return slip_violation

    def out_domain_flow(self):
        #ToDo: is this a usefull parameter, background is white so that is 'max velocity'
        _, mask = find_boundary(self.prediction)
        mask = ~mask
        masked = cv2.bitwise_and(self.prediction, self.prediction, mask=mask)
        cv2.imshow('masked', masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        flow_out_domain = 0
        return flow_out_domain

    def conservation(self):
        #ToDo: now we get pixel intensity difference, which corresponds to velocity
        # we should have in = out flow for every timestep, as assumed newtonian fluid?
        total_inlet = []
        total_outlet = []
        for image in self.timeseries:
            inlet, outlet = find_walls_inlet_outlet(image)
            inlet_values = [image[coord] for coord in inlet]
            outlet_values = [image[coord] for coord in outlet]
            total_inlet.append(sum(inlet_values))
            total_outlet.append(sum(outlet_values))


        net_flow = sum(total_inlet) - sum(total_outlet)
        return net_flow

    def comp_dom_errors(self):
        _, mask = find_boundary(self.image)
        masked_im = cv2.bitwise_and(self.image, self.image, mask=mask)
        masked_pred = cv2.bitwise_and(self.prediction, self.prediction, mask=mask)

        conv_im = convert_pixel_to_vel(masked_im, name='image')
        conv_pred = convert_pixel_to_vel(masked_pred, name = 'prediction')

        # Root MSE instead of MSE for interpretability
        RMSE = np.sqrt(((conv_im - conv_pred)**2).mean(axis=None))
        MAE = np.mean(np.abs(conv_im - conv_pred))
        ME = np.amax(np.abs(conv_im - conv_pred))

        return RMSE, MAE, ME


