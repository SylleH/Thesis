"""
Preprocessing functions for CFD simulations images to tensor input for model (datagenerators)
and for time serie windows to tensor input for Time models (data.datasets)

Author: Sylle Hoogeveen
"""

from utils.ops import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# scenario = "straight" #bifurcation, bend90 or branch
# data_dir = "data/"+scenario+"/"

def create_train_val_datagen(data_dir, batch_size, img_height, img_width):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2) #lots of options to specify
    train_generator = image_gen.flow_from_directory(data_dir,
                                                    target_size=(img_height,img_width),
                                                    color_mode="grayscale",
                                                    shuffle=True,
                                                    subset="training",
                                                    batch_size = batch_size,
                                                    class_mode="input",
                                                    seed = 123)

    val_generator = image_gen.flow_from_directory(data_dir,
                                                    target_size=(img_height,img_width),
                                                    color_mode="grayscale",
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



# def test():
#     train_generator, val_generator = create_train_val_datagen(data_dir, batch_size = 32,img_height = 856,img_width = 784)
#     img_batch, _ = train_generator.next()
#     first_image = img_batch[0]
#     print(first_image.shape)
#     plt.figure()
#     plt.imshow(first_image)
#     plt.show()

def create_TS_dataset(encoder, test_generator):
    """
    Function that encodes images supplied by test_generator with stored encoder and stores the encoded vectors
    in data.datasets which can be used as input for the Time models Every encoded vector is a training example,
    last encoded time cannot be used (no label available).

    :param encoder: trained encoder model
    :param test_generator: datagenerator containing images to be encoded
    :return window: instance of the class WindowGenerator (see ops.py),
                    containing train, validation and test datasets
    """

    #ToDo: encode all sequences, make sure they are NOT shuffled, then splitting in to train/val/test goes correctly
    data_NN_dir = 'data/test/'
    encoded = encoder.predict(test_generator)
    #print(encoded)
    TS_df = pd.DataFrame(encoded)
    label_column_names = ['feat '+str(x) for x in range(len(TS_df.columns))]
    TS_df.columns =  label_column_names

    #print(TS_df)
    #ToDo:   - concat with known parameters: timeserie characteristic,
    #           timestep, inlet velocity, outlet pressure and change in those
    #        - TS_df['timestep'] = timestep_list etc.
    timestep_list = []
    inlet_v_list = []
    outlet_p_list = []

    # Split the data in training, validation and test 70/20/10
    column_indices = {name: i for i, name in enumerate(TS_df.columns)}

    n = len(TS_df)
    train_df = TS_df[0:int(n * 0.7)]
    val_df = TS_df[int(n * 0.7):int(n * 0.9)]
    test_df = TS_df[int(n * 0.9):]

    num_features = TS_df.shape[1]
    window = WindowGenerator(input_width=1, label_width=1, shift=1,
                             train_df=train_df, val_df=val_df, test_df=test_df,
                             label_columns=label_column_names)
    return window
