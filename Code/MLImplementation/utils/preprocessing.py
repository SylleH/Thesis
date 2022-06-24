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
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.01) #lots of options to specify
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

def create_test_datagen(data_test_dir, batch_size, img_height, img_width):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # lots of options to specify
    test_generator = image_gen.flow_from_directory(data_test_dir,
                                                   target_size=(img_height, img_width),
                                                   color_mode="grayscale",
                                                   shuffle=False,
                                                   batch_size=batch_size,
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

def create_TS_dataset(data_dir, encoder, train_split, val_split, test):
    """
    Function that encodes images supplied by test_generator with stored encoder and stores the encoded vectors
    in data.datasets which can be used as input for the Time models Every encoded vector is a training example,
    last encoded time cannot be used (no label available).

    :param data_dir: directory to image data
    :param encoder: trained encoder model
    :param train_split, val_split: portion of dataset to go into train and validaiton dataset
    :param test: either True or False, if true al data in data_dit will be in test_ds
    :return window: instance of the class WindowGenerator (see ops.py),
                    returned to be able to show some specs
    :return train_ds, val_ds, test_ds: tf.data.Datasets for training, validation and testing
            if test = True train_ds and val_ds are None, if test= False test_ds is None
    """

    count  = 0

    for folder in os.listdir(data_dir):
        if folder == '.DS_Store':
            continue
        if not test:
            test_generator = create_test_datagen(data_dir+"/"+folder, 36, 192,256 )
        else:
            test_generator = create_test_datagen(data_dir, 36, 192, 256)

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

        column_indices = {name: i for i, name in enumerate(TS_df.columns)}

        n = len(TS_df)
        num_features = TS_df.shape[1]
        window = WindowGenerator(input_width=1, label_width=1, shift=1,
                             df=TS_df,label_columns=label_column_names)
        if count == 0:
            TS_ds = window.ds
        else:
            TS_ds = TS_ds.concatenate(window.ds)
        #print(tf.data.experimental.cardinality(TS_ds).numpy())
        count+=1

    ds_size = tf.data.experimental.cardinality(TS_ds).numpy()
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(TS_ds, ds_size, train_split=train_split,
                                                          val_split=val_split, test=test, shuffle=True)
    return window, train_ds, val_ds, test_ds
