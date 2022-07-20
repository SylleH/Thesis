from utils.preprocessing import *
from utils.tools import *
from model import E_TS_D
from utils.ops import *
import os
import yaml
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import date
import pydot
import pydotplus
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model




def trainer_AE(config, strategy, total=False):
    """
    Train Autoencoder network
    :param config: config dict loaded from config.yaml file
    :param strategy: training strategy determined by availability of GPUs
    :return history: model training history
    :return checkpoint_filepath: path to trained model
    """
    #import config, expand to see
    scenario = config["data"]["scenario"]
    model_dir = config["data"]["model_dir"]

    img_height = config["preprocess"]["img_height"]
    img_width = config["preprocess"]["img_width"]
    channels = config["preprocess"]["channels"]

    filters = config["AE"]["filters"]
    z_num = config["AE"]["z_num"]
    num_conv = config["AE"]["num_conv"]
    conv_k = config["AE"]["conv_k"]
    last_k = config["AE"]["last_k"]
    repeat = config["AE"]["repeat"]

    patience = config["AE"]["training"]["patience"]
    batch_size = config["AE"]["training"]["batch_size"]
    epochs = config["AE"]["training"]["epochs"]

    if total:
        node_num = config["TS"]["node_num"]
        num_layers = config["TS"]["num_layers"]
        dropout = config["TS"]["training"]["dropout"]

    #data_dir = "data/" + scenario + "/"
    data_dir = 'data/NN_testset/'
    #model_path = model_dir+"AE_"+str(scenario)+"_BS"+str(batch_size)+"_E"+str(epochs)+"_f"+str(filters)+"_z"+str(z_num)
    model_path = model_dir + "AE_straight_OVERFIT_val0.01_E126_f16_z150_oneTS"
    if total:
        model_path = model_dir+"total_"+str(scenario)+"_BS"+str(batch_size)+"_E"+str(epochs)+"_5steps"
    log_path = model_path + "/logs"
    checkpoint_filepath = model_path + "/checkpoint/"
    #fig_name = "loss_"+"AE_"+str(scenario)+"_BS"+str(batch_size)+"_E"+str(epochs)+"_f"+str(filters)+"_z"+str(z_num)
    fig_name = "loss" + "AE_straight_OVERFIT_val0.01_E126_f16_z150_oneTS"
    if total:
        fig_name = "loss_"+"total_"+str(scenario)+"_BS"+str(batch_size)+"_E"+str(epochs)+"_5steps"

    # define callbacks AE
    tb_callback = TensorBoard(log_dir=log_path)
    es_callback = EarlyStopping(patience= patience)
    modcheck_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'model_' + str(date.today())),
                                        save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)

    callbacks = [tb_callback, es_callback, modcheck_callback]

    if total:
        train_generator, val_generator = create_input_multi_output_gen("data/TS_seperately/", img_height, img_width,
                                                                       batch_size=batch_size, val_split=0.9)
    else:
        train_generator, val_generator = create_train_val_datagen(data_dir, batch_size, img_height, img_width)

    #create model and train with determined strategy
    with strategy.scope():
        img_inputs = keras.Input(shape=(img_height, img_width, channels)) #shape of each sample needs to be supplied, this is after cropping
        if not total:
            model = create_AE(img_inputs, filters=filters, z_num=z_num,
                                 repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)
            print(model.summary())
            history = train_and_store(model, train_generator, val_generator,
                                      batch_size, epochs, callbacks, fig_name)
        else:
            z, out, model = E_TS_D(img_inputs, filters, z_num, num_conv, conv_k, last_k, repeat,
                                    onum=150, num_layers=num_layers, node_num=node_num, dropout=dropout)
            tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            model.compile(loss="mse", optimizer="adam", metrics=["mse"])
            print(model.summary())

            #history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)

    return history, checkpoint_filepath


def trainer_TS(config, strategy, encoder):
    """
    Train TimeSeries network
    :param config: config dict loaded from config.yaml file
    :param strategy: training strategy determined by availability of GPUs
    :param encoder: encoder model needed to create latent vectors as training input to TS network
    :return history: model training history
    :return checkpoint_filepath: path to trained TimeSeries Network
    """
    #import config, expand to see
    scenario = config["data"]["scenario"]
    model_dir = config["data"]["model_dir"]

    w_input = config["TS"]["window"]["input"]
    w_shift = config["TS"]["window"]["shift"]
    w_label = config["TS"]["window"]["label"]

    network = config["TS"]["architecture"]
    num_layers = config["TS"]["num_layers"]
    nodenum = config["TS"]["node_num"]

    dropout = config["TS"]["training"]["dropout"]
    batch_size = config["TS"]["training"]["batch_size"]
    epochs = config["TS"]["training"]["epochs"]

    data_dir = "data/TS_seperately"
    model_path = model_dir + "TS_" + str(network) + "_" + str(scenario) + "_BS" + str(batch_size) + "_E" + str(
        epochs)+'_l'+str(num_layers)+'_n'+str(nodenum) +"_oneTS"
    log_path = model_path + "/logs"
    checkpoint_filepath = model_path + "/checkpoint/"
    fig_name = "loss" + "TS_" + str(network) + "_" + str(scenario) + "_BS" + str(batch_size) \
               + "_E" + str(epochs)+'_l'+str(num_layers)+'_n'+str(nodenum)

    # use encoded images to train NN
    window, train_ds, val_ds, test_ds, _,_ = create_TS_dataset(data_dir, encoder, w_input, w_shift, w_label,
                                                               batch_size=batch_size, train_split=0.9, val_split=0.1,
                                                               test=False)
    print(window)
    print('Input shape:', window.example[0].shape)

    # define callbacks TS
    tb_callback = TensorBoard(log_dir=log_path)
    es_callback = EarlyStopping(patience=25)
    modcheck_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'model_' + str(date.today())),
                                        save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
    callbacks = [tb_callback, es_callback, modcheck_callback]

    #create model and train with determined strategy
    with strategy.scope():
        encoded_inputs = keras.Input(shape=window.example[0].shape[1:])
        # print(encoded_inputs)

        # create_TS_network can take NN, RNN_LSTM and RNN_GRU as inputs for model
        TS_network = create_TS_network(x=encoded_inputs, onum=150, numlayers= num_layers,nodenum=nodenum, dropout=dropout, model='NN')
        print(TS_network.summary())

        history = train_and_store(TS_network, train_ds, val_ds, batch_size= batch_size, epochs=epochs,
                              callbacks=callbacks,fig_name=fig_name)

    return history, checkpoint_filepath