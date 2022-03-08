"""
Implementation inspired on Deep Fluids: A Generative Network for parameterized fluid simulations
using keras functional API
Author: Sylle Hoogeveen
"""



import tensorflow as tf
from tensorflow import keras
from keras import layers
from ops import *
import numpy as np



def Encoder(x, filters, z_num,  num_conv, conv_k, repeat, act=tf.nn.leaky_relu, name='encoder'):
    """
    Encoder network to obtain reduced dimension representation from velocity field input

    :param x: velocity field tensor with data format [Number of images in batch, Height, Width ,Channel_depth]
    :param filters: number of filters generated per convolutional layer
    :param z_num: latent dimension size (unsupervised part)
    :param num_conv: number of convolutional layers
    :param conv_k: kernel size
    :param repeat: number of repeats block of convolutional layers and skip connection
    :param act: activation function all layers

    :return z: reduced dimension representation
    :return encoder_model: keras model of encoder
    """

    x_shape = get_conv_shape(x)[1:] #returns list with [H,W,C]

    if repeat == 0:
        repeat_num = int(np.log2(np.max(x_shape[:-1])))-2 #nearest integer to log2(max(H,W))-3
    else:
        repeat_num = repeat

    ch = filters
    layer_num = 0
    x = layers.Conv2D(ch, kernel_size=conv_k,strides=1,activation=act, padding="same",name=str(layer_num)+'_conv')(x)
    # make copy for skip connection
    x0 = x

    layer_num+=1

    #BIG BLOCK
    for idx in range(repeat_num):

        #SMALL BLOCK
        for _ in range(num_conv):
            x = layers.Conv2D(filters, kernel_size=conv_k, strides=1, activation=act, padding="same", name=str(layer_num)+'_conv')(x)
            layer_num +=1

        ch += filters

        if idx < repeat_num-1:
            # skip connection
            x = layers.Concatenate(axis=-1, name=str(idx) + "_skip")([x, x0])  # concat on feature map = last dimension

            x = layers.MaxPool2D((2,2),padding="valid",name=str(idx)+"_pool")(x)
            #x = layers.Conv2D(ch, kernel_size=conv_k, strides=2, activation=act, padding="same",name=str(layer_num) + '_conv')(x)
            layer_num +=1
            x0=x

    flat = layers.Flatten()(x) #reshapes to flat tensor with same batch_size and dimension fitted to keep input size
    z = layers.Dense(z_num, activation=act, name=str(layer_num)+'_fc')(flat)

    encoder_model = keras.Model(x,z)
    return z, encoder_model


def Generator(z, filters, output_shape, num_conv, conv_k, last_k, repeat, act=tf.nn.leaky_relu, name='Generator'):
    """
    Generator (or decoder) network to obtain velocity field from reduced dimension representation

    :param z: (new) reduced dimension representation (z_t+1 = z_t + output NN)
    :param filters: number of filters generated per convolutional layer
    :param output_shape: velocity field dimensions [H,W,C]
    :param num_conv: number of convolutional layers
    :param conv_k: kernel size
    :param last_k: kernel size last convolutional layer, determines output size
    :param repeat: number of repeats block of convolutional layers and skip connection
    :param act: activation function all layers

    :return out: generated velocity field
    :return generator_model: keras model of generator
    """

    if repeat == 0:
        repeat_num = int(np.log2(np.max(output_shape[:-1]))) -2
    else:
        repeat_num = repeat

    x0_shape = [int(i/np.power(2, repeat_num-1)) for i in output_shape[:-1]] + [filters]#+(filters*repeat_num)]
    print('first layer:', x0_shape, 'to', output_shape)

    num_output = int(np.prod(x0_shape)) #number of output nodes flattend H*W*filters
    layer_num = 0

    x = layers.Dense(num_output, name= str(layer_num)+'_fc')(z)
    layer_num +=1

    #why not conv2DTranspose?
    x = layers.Reshape([x0_shape[0], x0_shape[1], x0_shape[2]])(x) #shape does not include batch size dimension
    x0=x

    #BIG BLOCK
    for idx in range(repeat_num):

        #SMALL BLOCK
        for _ in range(num_conv):
            x = layers.Conv2DTranspose(filters=filters, kernel_size=conv_k, strides=1, activation=act, padding="same",name=str(layer_num) + '_deconv')(x)
            layer_num += 1

        if idx < repeat_num - 1:
            #residual skip connection = element-wise sum of feature maps of input & output each convolutional layer block
            x = layers.Add(name=str(idx)+"_add")([x, x0])
            x = layers.UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear', name=str(idx)+"_upsample")(x)
            x0 = x

        else:
            x = layers.Add(name=str(idx)+"_add")([x, x0])

    out = layers.Conv2D(output_shape[-1], kernel_size=last_k, strides=1, padding="same", activation= keras.activations.relu, name=str(layer_num) + '_deconv')(x)

    generator_model = keras.Model(z,out)
    return out, generator_model

def AE(x, filters, z_num, num_conv, conv_k, last_k, repeat, act=tf.nn.leaky_relu, name='AE'):
    """
    Combination of Encoder and Generator networks

    :param: see encoder and generator parameters
    :return z: reduced representation learned by encoder
    :return out: generated velocity field from reduced representation
    """

    z,_ = Encoder(x, filters=filters, z_num=z_num, num_conv= num_conv, conv_k= conv_k, repeat= repeat, act=act)

    #z is new latent representation = z + output NN
    out,_ = Generator(z, filters=filters, output_shape = get_conv_shape(x)[1:],
                             num_conv=num_conv, conv_k=conv_k, last_k=last_k, repeat=repeat, act=act)
    autoencoder = keras.Model(x, out)

    return z, out, autoencoder

def LatentIntNN(x, onum, node_num=512, act = tf.nn.elu, dropout=0.1, name='LatentIntNN'):
    """
    Neural Network for time evolution in reduced dimension

    :param x: combination of z (reduced representation learned from encoder), p (known parameters) and
            delta_p (difference known parameters time t with t+1)
    :param nodenum: amount of nodes in second layer, first layer is nodenum*2 nodes
    :param onum: amount of nodes output layer, is dimension of c
    :param act: activation function for all layers
    :param dropout: dropout rate, chance a node is DROPPED (not kept)
    :returns delta_z: difference learned reduced representation time t with t+1
    """


    x = layers.BatchNormalization()(x)
    x = layers.Dense(node_num*2, activation=act)(x)
    x = layers.Dropout(dropout)(x) #keras automatically sets training to True only for training phase

    x = layers.BatchNormalization()(x)
    x = layers.Dense(node_num, activation=act)(x)
    x = layers.Dropout(dropout)(x)  # keras automatically sets training to True only for training phase

    delta_z = layers.Dense(onum)(x)
    return delta_z

def LatentIntLSTM():
    return 0