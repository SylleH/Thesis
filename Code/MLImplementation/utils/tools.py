"""
Functions for model creations, loading, training and evaluating
Author: Sylle Hoogeveen
"""


import tensorflow as tf
from tensorflow import keras
from utils.ops import *
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from model import AE, Time_NN, Time_LSTM, Time_GRU
import os
from datetime import date
import numpy as np
import cv2
import pandas as pd
import time





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


def train_and_store(model, train_generator,val_generator, epochs, callbacks,fig_name):
    # Note: model is stored by using the ModelCheckpoint callback
    history = model.fit(train_generator, validation_data = val_generator,
                          epochs= epochs,  callbacks = callbacks)

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
    predicted1, predicted2, predicted3, predicted4, predicted5, = model.predict(test_set)
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


def predict_one_beat(model, test_set, t, scenario, directory):
    pred_beats_folder = f"predicted_beats/{scenario}/{directory}"
    diff_image_folder = f"difference_images/{directory}"
    time_error_folder = f"error_timestep_plots/{directory}"

    for folder in [pred_beats_folder,diff_image_folder, time_error_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Directory {folder} made")

    beats = 1
    inputs, labels = test_set[0]
    U_df = pd.read_csv("inlet_Us/inlet_U_vp0.csv")
    Udif0 = U_df['U_0']
    Udif0 = Udif0[1:]
    print(len(Udif0))
    # im_inputs = inputs[0]
    # vel_inputs = inputs[1:]
    # Udif0, Udif1, Udif2, Udif3, Udif4, Udif5  = vel_inputs
    # #, Udif6, Udif7, Udif8, Udif9, Udif10
    # for i in range(4):
    #     Udif5 = np.append(Udif5, Udif1[i])
    #     Udif4 = np.append(Udif4, Udif5[-2])
    #     Udif3 = np.append(Udif3, Udif4[-2])
    #     Udif2 = np.append(Udif2, Udif3[-2])
    #     Udif1 = np.append(Udif1, Udif2[-2])
    #     Udif0 = np.append(Udif0, 0.08)


    n = int(100/t)
    print(n)
    c = n-1
    if t == 2:
        c2 = c-2
    elif t == 5:
        c2 = c
    elif t == 1:
        c = 95
        #c = 99
        c2 = 95
    else:
        c2 = c-1

    n=100
    predicted_beat = np.empty([n, 192, 256, 1])
    true_beat = np.empty([n, 192, 256, 1])
    label1, label2, label3, label4, label5= labels
    #,label6, label7, label8, label9, label10
    error_graph = []
    predicted, predicted2, predicted3, predicted4, predicted5= model.predict(test_set)
    #,predicted6, predicted7, predicted8, predicted9, predicted10
    #predicted_beat[0] = predicted[0] #this is t=1
    predicted = predicted[0]

    t_begin = time.perf_counter()
    for b in range(beats):
        idx = (b * 100)
        predicted_beat[idx] = predicted  # this is t=1, t=101, t=201, or predicted[0], predicted[100], predicted[200]
        true_beat[idx] = label1[0]
        if idx == 0:
            error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[0], axis=0),
                                            prediction=np.expand_dims(predicted_beat[0], axis=0),
                                            scenario=scenario))  # must expand dims as the first entry is only one image
        else:
            error_graph.append(
                ErrorMetrics(images=np.expand_dims(true_beat[idx], axis=0),
                                            prediction=np.expand_dims(predicted_beat[idx], axis=0),
                             scenario=scenario))

        for i in range(95): #99 for t=1, 49 for t=2, 32 for t=3, 24 for t=4, 19 for t=5
            idx = (b * 100) + i
            inputs = [predicted_beat[idx]]#, Udif0[i + 1], Udif1[i + 1], Udif2[i + 1], Udif3[i + 1], Udif4[i + 1], Udif5[i + 1]]# , Udif6[i + 1], Udif7[i + 1], Udif8[i + 1], Udif9[i + 1], Udif10[i + 1]]
            inputs = [np.expand_dims(imvel, axis=0) for imvel in inputs]
            imvel = tuple(inputs)
            predicted, predicted2, predicted3, predicted4, predicted5 = model.predict(imvel)
                #,predicted6, predicted7, predicted8, predicted9, predicted10
            predicted_beat[idx + 1, :, :, :] = predicted
            if i in range(c2): #47 for t=2, 31 for t=3, 23 for t=4, 19 for t=5, 95 for t=1
                true_beat[idx + 1, :, :, :] = label1[(i + 1)*t]
                error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[idx+1], axis=0),
                                            prediction=np.expand_dims(predicted_beat[idx+1], axis=0), scenario=scenario))
            # if t == 2:
            #         true_beat[-2] = label3[-1]
        if t == 1: # and i == 95:
            true_beat[-4] = label2[-1]
            predicted_beat[-4] = predicted2[-1]
            error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[96], axis=0),
                                            prediction=np.expand_dims(predicted_beat[96], axis=0), scenario=scenario))
            #elif t == 1 and i == 96:
            true_beat[-3] = label3[-1]
            predicted_beat[-3] = predicted3[-1]
            error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[97], axis=0),
                                            prediction=np.expand_dims(predicted_beat[97], axis=0), scenario=scenario))
            #elif t == 1 and i == 97:
            true_beat[-2] = label4[-1]
            predicted_beat[-2] = predicted4[-1]
            error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[98], axis=0),
                                            prediction=np.expand_dims(predicted_beat[98], axis=0), scenario=scenario))
            # true_beat[-6] = label5[-1]
            # predicted_beat[-6] = predicted5[-1]
            # error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[94], axis=0),
            #                                 prediction=np.expand_dims(predicted_beat[94], axis=0), scenario=scenario))
            # true_beat[-5] = label6[-1]
            # predicted_beat[-5] = predicted6[-1]
            # error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[95], axis=0),
            #                                 prediction=np.expand_dims(predicted_beat[95], axis=0), scenario=scenario))
            # true_beat[-4] = label7[-1]
            # predicted_beat[-4] = predicted7[-1]
            # error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[96], axis=0),
            #                                 prediction=np.expand_dims(predicted_beat[96], axis=0), scenario=scenario))
            # true_beat[-3] = label8[-1]
            # predicted_beat[-3] = predicted8[-1]
            # error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[97], axis=0),
            #                                 prediction=np.expand_dims(predicted_beat[97], axis=0), scenario=scenario))
            # true_beat[-2] = label9[-1]
            # predicted_beat[-2] = predicted9[-1]
            # error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[98], axis=0),
             #                               prediction=np.expand_dims(predicted_beat[98], axis=0), scenario=scenario))
            print('True')
        if t in [1,2,3,4]: # and i == 98:
            true_beat[-1] = label5[-1]
            predicted_beat[-1] = predicted5[-1]
            error_graph.append(ErrorMetrics(images=np.expand_dims(true_beat[94+5], axis=0),
                                        prediction=np.expand_dims(predicted_beat[94+5], axis=0), scenario=scenario))
            print('True')
        # if beats != 1 and b < beats:
        #     #print(idx+1)
        #     inputs = [predicted_beat[idx+1], Udif0[0], Udif1[0], Udif2[0], Udif3[0], Udif4[0], Udif5[0]] #this is t=100, t=200, t=300 as input or predicted[99], predicted[199]
        #     inputs = [np.expand_dims(imvel, axis=0) for imvel in inputs]
        #     imvel = tuple(inputs)
        #     predicted, predicted2, predicted3, predicted4, predicted5 = model.predict(imvel)
    t_end = time.perf_counter()
    print(f"Time to predict {beats} heartbeats is {t_end - t_begin:0.4f} seconds")
    #print(len(error_graph))


    all_timesteps_errors = ErrorMetrics(images = true_beat, prediction=predicted_beat, scenario=scenario)

    # for i in range(len(error_graph)): #100/t
    #     plt.figure()
    #     plt.imshow(predicted_beat[i])
    #     plt.axis('off')
    #     plt.gray()
    #     plt.savefig(f"predicted_beats/bifurcation/{directory}/new_{scenario}_v_predicted_beat_{i}.png",bbox_inches='tight',pad_inches = 0)
    #     #plt.show()
    #     plt.close()
    #
    # avg_slip_mean_rel, avg_slip_mean_abs, max_slip_max_rel, max_slip_max_abs, avg_RMSE, avg_RRMSE, \
    # avg_MAE_rel, avg_MAE_abs, max_ME_rel, max_ME_abs, df_max_abs, df_max_rel = all_timesteps_errors.evaluate()
    # net_flow_im = all_timesteps_errors.conservation(pred=False, ts=t)
    # net_flow_pred = all_timesteps_errors.conservation(pred=True, ts=t)
    # net_flow_error = np.abs(net_flow_im - net_flow_pred)
    # print(f"boundary MAE rel: {avg_slip_mean_rel} \n"
    #       f"domain MAE rel: {avg_MAE_rel} \n"
    #       f"boundary ME rel: {max_slip_max_rel} \n"
    #       f"domain ME rel: {max_ME_rel} \n"
    #       f"domain RRMSE: {avg_RRMSE} \n"
    #       f"boundary MAE abs: {avg_slip_mean_abs} \n"
    #       f"domain MAE abs: {avg_MAE_abs} \n"
    #       f"boundary ME abs: {max_slip_max_abs} \n"
    #       f"domain ME abs: {max_ME_abs} \n"
    #       f"domain RMSE: {avg_RMSE} \n"
    #       f"netflow error: {net_flow_error} \n")
    # plt.rcParams['font.size'] = '18'
    # fig, ax = plt.subplots(figsize=(2, 10))
    # fig.subplots_adjust(right=0.3)
    #
    # cmap = matplotlib.cm.OrRd
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max([max_slip_max_abs, max_ME_abs]))
    #
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              cax=ax, orientation='vertical', label='error [m/s]')
    # plt.tight_layout()
    # fig.savefig(f"colorbar_{directory}.jpeg")
    # all_timesteps_errors.difference_images(np.max([max_slip_max_abs, max_ME_abs]))
    #
    # contour,mask = find_boundary(predicted_beat[0])
    # make_gif('predicted_beats/bifurcation',directory, mask)

    # df_max_abs['y_loc'] = -1 * df_max_abs['y_loc']
    # abs = sns.relplot(data=df_max_abs, x='x_loc', y='y_loc', hue='group', size='count')
    # abs.fig.suptitle('predicted beat max absolute error')
    # abs.savefig('predicted beat max absolute error.png')
    # df_max_rel['y_loc'] = -1 * df_max_rel['y_loc']
    # rel = sns.relplot(data=df_max_rel, x='x_loc', y='y_loc', hue='group', size='count')
    # rel.fig.suptitle('predicted beat max relative error')
    # rel.savefig('predicted beat max relative error.png')



    avg_slip_mean_rel = np.empty([len(error_graph)])
    avg_slip_mean_abs = np.empty([len(error_graph)])
    max_slip_max_rel = np.empty([len(error_graph)])
    max_slip_max_abs = np.empty([len(error_graph)])
    avg_RMSE = np.empty([len(error_graph)])
    avg_RRMSE = np.empty([len(error_graph)])
    avg_MAE_rel = np.empty([len(error_graph)])
    avg_MAE_abs = np.empty([len(error_graph)])
    max_ME_rel = np.empty([len(error_graph)])
    max_ME_abs = np.empty([len(error_graph)])
    #
    #
    for idx, errors in enumerate(error_graph):
        avg_slip_mean_rel[idx], avg_slip_mean_abs[idx], max_slip_max_rel[idx], max_slip_max_abs[idx], avg_RMSE[idx], \
        avg_RRMSE[idx],avg_MAE_rel[idx], avg_MAE_abs[idx], max_ME_rel[idx], max_ME_abs[idx], _ ,_ = errors.evaluate()
        print(idx)

    #Udif0_3b = np.concatenate((Udif0,Udif0,Udif0), axis=0)


    plt.rcParams['font.size'] = '18'
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),avg_slip_mean_abs, 'r')
    ax2.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),Udif0, 'blue')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('error [m/s]', color = 'r')
    ax2.set_ylabel('inlet velocity [m/s]', color = 'blue')
    ax2.set_ylim([0, 0.5])
    ax1.set_xlim([0,len(error_graph)/100])
    plt.title("boundary MAE absolute")
    plt.tight_layout()
    plt.savefig(f"error_timestep_plots/{directory}/boundary MAE absolute")
    plt.show()


    plt.rcParams['font.size'] = '18'
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),avg_slip_mean_rel, 'r')
    ax2.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),Udif0, 'blue')
    plt.title("boundary MAE relative")
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('error', color ='r')

    ax2.set_ylabel('inlet velocity [m/s]', color='blue')
    ax2.set_ylim([0, 0.5])
    ax1.set_xlim([0,len(error_graph)/100])
    plt.tight_layout()
    plt.savefig(f"error_timestep_plots/{directory}/boundary MAE relative")
    plt.show()

    # plt.figure()
    # plt.plot(np.linspace(1,97,97),max_slip_max_abs)
    # plt.title("boundary ME")
    # plt.savefig(scenario+" boundary ME")
    # plt.show()

    plt.rcParams['font.size'] = '18'
    fig, ax1 = plt.subplots()
    ax1.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),avg_MAE_abs, 'r')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('error [m/s]', color = 'r')
    ax1.set_xlim([0,len(error_graph)/100])
    ax1.set_ylim([0, 0.62])
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),Udif0, 'blue')
    ax2.set_ylabel('inlet velocity [m/s]', color = 'blue')
    ax2.set_ylim([0, 0.5])
    plt.title("domain MAE absolute")
    plt.tight_layout()
    plt.savefig(f"error_timestep_plots/{directory}/domain MAE absolute")
    plt.show()


    plt.rcParams['font.size'] = '18'
    fig, ax1 = plt.subplots()
    ax1.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),avg_MAE_rel, 'r')
    ax1.set_ylabel('error', color ='r')
    ax1.set_xlabel('time [s]')
    ax1.set_ylim([0, 1.25])
    ax1.set_xlim([0,len(error_graph)/100])
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0.01,len(error_graph)/100,len(error_graph)),Udif0, 'blue')
    ax2.set_ylabel('inlet velocity [m/s]', color = 'blue')
    ax2.set_ylim([0, 0.5])
    plt.title("domain MAE relative")
    plt.tight_layout()
    plt.savefig(f"error_timestep_plots/{directory}/domain MAE relative")
    plt.show()

    # plt.figure()
    # plt.plot(np.linspace(1,97,97),max_ME_abs)
    # plt.title("domain ME")
    # plt.savefig(scenario+" domain ME")
    # plt.show()

def load_model(network, checkpoint_filepath, date=str(date.today())):
    checkpoint_filepath = 'model/' + checkpoint_filepath
    print(network)
    print(checkpoint_filepath)
    print(date)

    if os.path.exists(os.path.join(checkpoint_filepath, 'model_' + date)):
        model = keras.models.load_model(os.path.join(checkpoint_filepath, 'model_' + date), compile=False)
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

    def __init__(self, images, prediction, scenario):
        self.images = images
        self.predictions = prediction
        self.scenario = scenario

        gray = cm.get_cmap('gray')
        self.colormap_lookup = gray(range(256))[:,0]

    def difference_images(self, diff_max):
        print(len(self.images))
        print(len(self.predictions))


        for i in range(len(self.images)):
            self.image = self.images[i]
            self.prediction = self.predictions[i]
            contour, mask = find_boundary(self.images[0])
            mask_inv = ((~mask / 255).astype('float64'))
            condition = np.stack((mask,) * 4, axis=-1) > 0.5

            masked_im = convert_pixel_to_vel(cv2.bitwise_and(self.image, self.image, mask=mask))
            masked_pred = convert_pixel_to_vel(cv2.bitwise_and(self.prediction, self.prediction, mask=mask))
            #roi = np.nonzero(mask)
            difference = np.abs(masked_im - masked_pred)

            #mask_inv = ((~mask)/255).astype(np.float64)
            # segment = cv2.add(mask_inv[roi],difference[roi])
            # mask_inv[roi] = np.squeeze(segment)
            image = (difference/diff_max) #scale difference to 0-1 by max difference
            #print(np.max(image))

            im = cm.OrRd(image)

            #print(np.max(im))
            #image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
            #output_image = np.where(condition, image, np.stack((mask_inv,) * 3, axis=-1))

            output_image = np.where(condition, im, np.stack((mask_inv,)*4, axis=-1))

            fig, ax = plt.subplots()
            #im = ax.imshow(image, cmap='viridis', vmin=0, vmax=diff_max)
            #fig.colorbar(im)
            ax.imshow(output_image)
            ax.axis('off')
            fig.savefig(f"difference_images/oldmodel_bend/{self.scenario}_diff_{i}", bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        folder_path = 'difference_images/oldmodel_bend'
        make_gif_diff(folder_path)

    def noslip(self):
        # image = cv2.imread('../data/NN_testset/straight/straight_w11.0_v3.5_2000.png', cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(src=image, dsize=(256,192))
        side, _, _ = find_walls_inlet_outlet(self.image, scenario=self.scenario)

        # contour, thresh = find_boundary(self.image)
        # mask = np.zeros(self.image.shape, dtype='uint8')
        # mask = cv2.drawContours(image=mask, contours=[contour], contourIdx=-1, color=255,
        #                   thickness=1, lineType=cv2.LINE_AA)
        # ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        prediction = convert_im_to_256int(self.prediction)
        image = convert_im_to_256int(self.image)

        # result_im = cv2.bitwise_and(image, image, mask = mask)
        # result_pred = cv2.bitwise_and(prediction, prediction, mask=mask)
        # cv2.imshow('mask', mask)
        # cv2.imshow('contour in binary im', result_im)
        # cv2.imshow('contour in binary pred', result_pred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        true_side_values = [image[coord] for coord in side]
        side_values = [prediction[coord] for coord in side]

        #convert from pixel color error to velocity error
        true_side_values = np.array([self.colormap_lookup[value[0]]*0.5 for value in true_side_values])
        side_values = np.array([self.colormap_lookup[value[0]]*0.5 for value in side_values])
        # print(f"side outlier coord: {side[0]} and {side[-1]}")
        # print(f"true outlier values: {true_side_values[0]} and {true_side_values[-1]}")
        # print(f"predicted outlier values: {side_values[0]} and {side_values[-1]}")

        side_rel_error = np.zeros_like(true_side_values)
        non_zero_idx = np.nonzero(true_side_values)[0].tolist()
        for idx, value in enumerate(true_side_values):
            if idx in non_zero_idx:
                side_rel_error[idx] = np.divide(np.abs(side_values[idx] - value), value)
            else:
                side_rel_error[idx] = side_values[idx]

        side_error = np.abs(np.subtract(true_side_values,side_values))

        side_mean_rel = np.mean(side_rel_error)
        side_mean_abs = np.mean(side_error)
        side_max_rel = np.amax(side_rel_error)
        side_max_abs = np.amax(side_error)
        max_idx_abs = np.argmax(side_error)
        max_idx_rel = np.argmax(side_rel_error)
        loc_max_abs = side[max_idx_abs]
        loc_max_rel = side[max_idx_rel]
        coord_max_abs = (loc_max_abs[1], loc_max_abs[0])
        coord_max_rel = (loc_max_rel[1], loc_max_rel[0])
        # print(f"mean slip: {side_mean_abs}")
        # print(f"max slip error: {side_max_abs}")
        # print(f"max coord: {coord_max}")
        return side_mean_rel, side_mean_abs, side_max_rel, side_max_abs, coord_max_abs, coord_max_rel

    def conservation(self, pred, ts):
        total_inlet = []
        total_outlet = []

        if pred == True:
            fields = self.predictions
        else:
            fields = self.images

        for i in range(len(fields)):
            image = fields[i]
            _, inlet, outlet = find_walls_inlet_outlet(self.images[i], scenario=self.scenario)

            image = convert_im_to_256int(image)

            inlet_values = [image[coord] for coord in inlet]
            outlet_values = [image[coord] for coord in outlet]

            #convert pixel intensity to velocity
            inlet_v = [self.colormap_lookup[value[0]] * 0.5 for value in inlet_values]
            outlet_v = [self.colormap_lookup[value[0]]*0.5 for value in outlet_values]

            #convert velocity to volume of flow (135 pixels representing 22mm -> 0.00016m )
            if self.scenario == 'straight':
                fp = 0.00016
            elif self.scenario == 'bend':
                fp = 0.00043
            elif self.scenario == 'bifurcation':
                fp = 0.00017
            else:
                print('netflow not defined')
                exit(code=3)
            inlet_flow = [vel*fp*0.01*ts for vel in inlet_v]
            outlet_flow = [vel*fp*0.01*ts for vel in outlet_v]
            total_inlet.append(sum(inlet_flow))
            total_outlet.append(sum(outlet_flow))

        # print(total_inlet)
        # print(total_outlet)
        net_flow = sum(total_inlet) - sum(total_outlet)
        return net_flow

    def comp_dom_errors(self):
        contour, mask = find_boundary(self.image)
        #remove boundary
        if self.scenario == 'straight':
            mask[:,195:196] = 0
            mask[:,60] = 0
            mask[4:5,:] = 0
            mask[186,:] = 0
        else:
            cv2.drawContours(mask, [contour], -1, 0, 1)

        masked_im = cv2.bitwise_and(self.image, self.image, mask=mask)
        masked_pred = cv2.bitwise_and(self.prediction, self.prediction, mask=mask)

        # cv2.imshow('image', masked_im)
        # cv2.imshow('prediction', masked_pred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        conv_im = convert_pixel_to_vel(masked_im)
        conv_pred = convert_pixel_to_vel(masked_pred)

        # Root MSE instead of MSE for interpretability
        non_zero_idx = np.nonzero(conv_im)
        RMSE = np.sqrt((np.power(np.subtract(conv_im, conv_pred),2)).mean(axis=None))
        RRMSE = np.sqrt(np.divide((np.power(np.subtract(conv_im,conv_pred),2)).mean(axis=None), (np.power(conv_im,2)).sum(axis=None)))

        RAE = np.ones_like(conv_im) * np.abs(np.subtract(conv_im,conv_pred))
        abs_ME = np.amax(RAE)
        loc_max_abs = np.unravel_index(np.argmax(RAE), shape=(192,256))
        coord_max_abs = (loc_max_abs[1], loc_max_abs[0])
        abs_MAE = RAE.mean(axis=None)
        RAE[non_zero_idx] = np.divide(RAE[non_zero_idx], conv_im[non_zero_idx])
        #RRMSE = np.sqrt((np.power(RAE, 2)).mean(axis=None))

        loc_max_rel = np.unravel_index(np.argmax(RAE), shape=(192,256))
        coord_max_rel = (loc_max_rel[1], loc_max_rel[0])
        # print(f"max domain error: {abs_ME}")
        # print(f"max domain coord: {max_coord}")

        rel_MAE = np.mean(RAE)
        rel_ME = np.amax(RAE)

        return RMSE, RRMSE, abs_MAE, rel_MAE, rel_ME, abs_ME, coord_max_abs, coord_max_rel

    def evaluate(self):
        rel_slip_mean = []
        abs_slip_mean = []
        rel_slip_max = []
        abs_slip_max = []
        RMSE_list = []
        RRMSE_list = []
        rel_MAE_list = []
        abs_MAE_list = []
        rel_ME_list = []
        abs_ME_list = []
        df_b_max_rel = pd.DataFrame(columns=['x_loc', 'y_loc', 'group', 'count'])
        df_b_max_abs = pd.DataFrame(columns=['x_loc', 'y_loc', 'group', 'count'])
        df_d_max_rel = pd.DataFrame(columns=['x_loc', 'y_loc', 'group', 'count'])
        df_d_max_abs = pd.DataFrame(columns=['x_loc', 'y_loc', 'group', 'count'])

        for i in range(len(self.images)):
            self.image = self.images[i]
            self.prediction = self.predictions[i]

            side_mean_rel, side_mean_abs, side_max_rel, side_max_abs, \
            coord_max_abs, coord_max_rel = self.noslip()
            rel_slip_mean.append(side_mean_rel)
            abs_slip_mean.append(side_mean_abs)
            rel_slip_max.append(side_max_rel)
            abs_slip_max.append(side_max_abs)
            df_b_max_rel.at[i,'x_loc']= coord_max_rel[0]
            df_b_max_rel.at[i,'y_loc']= coord_max_rel[1]
            df_b_max_abs.at[i,'x_loc']= coord_max_abs[0]
            df_b_max_abs.at[i,'y_loc']= coord_max_abs[1]

            RMSE, RRMSE, abs_MAE, rel_MAE, rel_ME, abs_ME, \
            coord_max_abs, coord_max_rel = self.comp_dom_errors()
            RMSE_list.append(RMSE)
            RRMSE_list.append(RRMSE)
            rel_MAE_list.append(rel_MAE)
            abs_MAE_list.append(abs_MAE)
            rel_ME_list.append(rel_ME)
            abs_ME_list.append(abs_ME)
            df_d_max_rel.at[i,'x_loc']= coord_max_rel[0]
            df_d_max_rel.at[i,'y_loc']= coord_max_rel[1]
            df_d_max_abs.at[i,'x_loc']= coord_max_abs[0]
            df_d_max_abs.at[i,'y_loc']= coord_max_abs[1]

        df_b_max_rel = df_b_max_rel.groupby(['x_loc', 'y_loc']).size().reset_index(name='count')
        df_b_max_rel['group'] = 'boundary'
        df_b_max_abs = df_b_max_rel.groupby(['x_loc', 'y_loc']).size().reset_index(name='count')
        df_b_max_abs['group'] = 'boundary'
        df_d_max_rel = df_d_max_rel.groupby(['x_loc', 'y_loc']).size().reset_index(name='count')
        df_d_max_rel['group'] = 'domain'
        df_d_max_abs = df_d_max_rel.groupby(['x_loc', 'y_loc']).size().reset_index(name='count')
        df_d_max_abs['group'] = 'domain'
        df_max_rel = pd.concat([df_b_max_rel, df_d_max_rel], axis = 0, ignore_index=True)
        df_max_abs = pd.concat([df_b_max_abs, df_d_max_abs], axis=0, ignore_index=True)

        avg_slip_mean_rel = np.mean(np.array(rel_slip_mean))
        avg_slip_mean_abs = np.mean(np.array(abs_slip_mean))
        max_slip_max_rel = np.amax(np.array(rel_slip_max))
        max_slip_max_abs = np.amax(np.array(abs_slip_max))
        avg_RMSE = np.mean(np.array(RMSE_list))
        avg_RRMSE = np.mean(np.array(RRMSE_list))
        avg_MAE_rel = np.mean(np.array(rel_MAE_list))
        avg_MAE_abs = np.mean(np.array(abs_MAE_list))
        max_ME_rel = np.amax(np.array(rel_ME_list))
        max_ME_abs = np.amax(np.array(abs_ME_list))

        return avg_slip_mean_rel, avg_slip_mean_abs, max_slip_max_rel, max_slip_max_abs, avg_RMSE,\
               avg_RRMSE, avg_MAE_rel, avg_MAE_abs, max_ME_rel, max_ME_abs, df_max_abs, df_max_rel


# images = np.zeros([100, 192,256,1])
# predictions = np.zeros([100, 192,256,1])
# image_list = [file for file in sorted(glob.glob(f"../../DataGeneration/Data_Generated/Grayscale/Bend/bend_w16.0_v5.0/bend/*.png"))]
#
# for i in range(100):
#     images[i] = np.expand_dims(np.flip(cv2.resize(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), [256,192]), axis=0),axis=-1)/255
#     predictions[i] = np.expand_dims(cv2.resize(cv2.imread(f"../predicted_beats/bend/LSTM_t1_predicted_beat_{i}.png", cv2.IMREAD_GRAYSCALE), [256,192]), axis=-1)/255
#
# errors = ErrorMetrics(images=images, prediction=predictions, scenario='bend')
# errors.difference_images(0.18)