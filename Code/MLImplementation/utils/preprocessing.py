"""
Preprocessing functions for CFD simulations images to tensor input for model (datagenerators)
and for time serie windows to tensor input for Time models (data.datasets)

Author: Sylle Hoogeveen
"""

from utils.ops import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2
#import hypertools as hyp

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


def create_input_multi_output_gen(data_dir, img_height, img_width, batch_size, val_split, test=False):
    """
    Function to create train and validation image generator for 1 input, 5 outputs model
    Note: if all data is loaded at once instead of using a generator, Out Of Memory error van occur
    :param data_dir:
    :param img_height:
    :param img_width:
    :param batch_size:
    :param val_split:
    :param test:
    :return train_generator:
    :return val_generator
    """

    #ToDo: hardcoded 1 input 5 outputs, make flexible
    input_files = []
    out1_files = []
    out2_files = []
    out3_files = []
    out4_files = []
    out5_files = []

    for dir in os.listdir(data_dir):
        if dir == ".DS_Store":
            continue
        if test:
            path = data_dir+"/straight/"
            files = os.listdir(path)
        else:
            path = data_dir+dir+"/straight/"
            files = os.listdir(path)
        if '.DS_Store' in files: files.remove('.DS_Store')
        files_sorted = sorted(files)
        #print(files_sorted)
        label_files= files_sorted.copy()
        label_files.pop(0)

        for i in range(5):
             files_sorted.pop(-1)


        for i in range(len(files_sorted)):
            input_files.append(path+files_sorted[i])
            out1_files.append(path+label_files[i])
            out2_files.append(path+label_files[i+1])
            out3_files.append(path+label_files[i+2])
            out4_files.append(path+label_files[i+3])
            out5_files.append(path+label_files[i+4])

    df_in = pd.DataFrame(input_files)
    df_out1 = pd.DataFrame(out1_files)
    df_out2 = pd.DataFrame(out2_files)
    df_out3 = pd.DataFrame(out3_files)
    df_out4 = pd.DataFrame(out4_files)
    df_out5 = pd.DataFrame(out5_files)
    df_all = pd.concat([df_in, df_out1, df_out2, df_out3, df_out4, df_out5], axis=1)
    for df in [df_in, df_out1, df_out2, df_out3, df_out4, df_out5]:
        df.columns = ['filename']

    df_all.columns = ['filename', 'out1', 'out2', 'out3', 'out4', 'out5']
    df_train = df_all.sample(frac=val_split)
    df_val = df_all.drop(df_train.index)

    train_generator = CustomDataGen(df_train, X_col={'path': 'filename'},
                              y_col={'out1': 'out1','out2': 'out2','out3': 'out3','out4': 'out4',
                                     'out5': 'out5'}, batch_size=batch_size, input_size=(img_height,img_width))

    val_generator = CustomDataGen(df_val, X_col={'path': 'filename'},
                              y_col={'out1': 'out1','out2': 'out2','out3': 'out3','out4': 'out4',
                                     'out5': 'out5'}, batch_size=batch_size, input_size=(img_height,img_width))

    return train_generator, val_generator

# def test():
#     train_generator, val_generator = create_train_val_datagen(data_dir, batch_size = 32,img_height = 856,img_width = 784)
#     img_batch, _ = train_generator.next()
#     first_image = img_batch[0]
#     print(first_image.shape)
#     plt.figure()
#     plt.imshow(first_image)
#     plt.show()

def create_TS_dataset(data_dir, encoder, input, shift, label, batch_size,train_split, val_split, test):
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
    # v_df = pd.read_csv('inlet_v.csv')
    # p_df = pd.read_csv('outlet_p.csv')

    for folder in os.listdir(data_dir):
        print(folder)
        if folder == '.DS_Store':
            continue
        if not test:
            test_generator = create_test_datagen(data_dir+"/"+folder, 101, 192,256 )
            shuffle = True
        else:
            test_generator = create_test_datagen(data_dir, 101, 192, 256)
            shuffle = False

        encoded = encoder.predict(test_generator)
        TS_df = pd.DataFrame(encoded)

        #ToDo: USE ONLY THESE AS LABEL COLUMNS or also time, velocity, pressure?
        label_column_names = ['feat '+str(x) for x in range(len(TS_df.columns))]
        TS_df.columns = label_column_names
        #label_column_names = label_column_names + ['time', 'velocity', 'pressure']
        #print(TS_df)

        # Add known inlet velocity, outlet pressure and time step to dataframe
        # scale pressure so that it is in same order of magnitude as other features
        #TS_v_df = pd.concat([TS_df, v_df, p_df['pressure']/10], axis=1)
        #print(TS_v_df)
        num_features = TS_df.shape[1]

        # for j in range(30):
        #     plt.figure(j)
        #     for i in range(5):
        #         plt.plot(TS_v_df['time'],TS_v_df.iloc[:,i+(5*j)], label=label_column_names[i+(5*j)])
        #     plt.plot(TS_v_df['time'], TS_v_df['velocity'],label='velocity')
        #     plt.plot(TS_v_df['time'], TS_v_df['pressure'], label='pressure')
        #     plt.legend(loc="upper right")
        #     plt.title(str(folder))
        #     plt.show()


        #print(TS_v_df)

        #column_indices = {name: i for i, name in enumerate(TS_df.columns)}

        n = len(TS_df)
        window = WindowGenerator(input_width=input, label_width=label, shift=shift,
                             df=TS_df,batchsize=batch_size,label_columns=label_column_names, shuffle=shuffle)
        if count == 0:
            TS_ds = window.ds
        else:
            TS_ds = TS_ds.concatenate(window.ds)
        #print(TS_v_ds)
        #print(tf.data.experimental.cardinality(TS_v_ds).numpy())
        count+=1

    ds_size = tf.data.experimental.cardinality(TS_ds).numpy()
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(TS_ds, ds_size, train_split=train_split,
                                                          val_split=val_split, test=test, shuffle=shuffle)
    return window, train_ds, val_ds, test_ds, test_generator, encoded
