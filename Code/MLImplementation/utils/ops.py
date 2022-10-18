"""
Funtions needed at some point in other python files

Author: Sylle Hoogeveen
"""

import os
#from PIL import Image
import tensorflow as tf
import numpy as np
import yaml
import cv2
import glob
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import statistics
from statistics import mode
import random
import imageio
from skimage import exposure
from keras import backend as K

class CustomDataGen(tf.keras.utils.Sequence):
    """
    Custom Image data generator for multiple inputs and outputs, adapted from:
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    """

    def __init__(self, df, X_col, y_col, batch_size, input_size, augmentation=True, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.augmentation = augmentation
        self.shuffle = shuffle

        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __data_augmentation(self, img, target_size, random_shift, random_flip, low_vel, label):
        x_shift = int(target_size[1] * random_shift)

        #ToDo: train with alternative, no need for flip :-)
        # result = np.empty_like(img)
        # if x_shift > 0:
        #     result[:x_shift] = 255
        #     result[x_shift:] = img[:-x_shift]
        # elif x_shift < 0:
        #     result[x_shift:] = 255
        #     result[:x_shift] = img[-x_shift:]
        # else:
        #     result[:] = img

        # for i in range(target_size[0]-1, target_size[0] - x_shift, -1):
        #     img = np.roll(img, -1, axis=1)
        #     img[:,-1] = 255
        if not label: #low_vel and
            # matplotlib.pyplot.imshow(img, cmap='gray')
            # matplotlib.pyplot.show()
            img_eq = exposure.equalize_hist(img)
            img_adapteq = exposure.equalize_adapthist(img)
            img_log = exposure.adjust_log(img)
            # matplotlib.pyplot.imshow(img_eq, cmap='gray')
            # matplotlib.pyplot.show()
            # matplotlib.pyplot.imshow(img_adapteq, cmap='gray')
            # matplotlib.pyplot.show()
            # matplotlib.pyplot.imshow(img_log, cmap ='gray')
            # matplotlib.pyplot.show()
            img = img_eq

        if random_flip == 1:
            img = np.flip(img, axis = 1)
        return img

    def __get_image_input(self, path, target_size, shift, flip, low_vel, label):
        image = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy() / 255.

        if self.augmentation:
            image_arr = self.__data_augmentation(image_arr, target_size, shift, flip, low_vel, label)
        #image_arr = np.flip(image_arr, axis=0)
        return image_arr

    def __get_vel_input(self, vel):
        return vel

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        in_batches = [[] for i in range(len(self.X_col))]
        #in_names = [f"in{x}" for x in range(len(self.X_col))]
        in_names = ['in0'] + [f"vel{x}" for x in range(len(self.X_col)-1)]
        for i in range(len(self.X_col)):
            in_batches[i] = batches[self.X_col[in_names[i]]]

        out_batches = [[] for i in range(len(self.y_col))]
        out_names = [f"out{x}" for x in range(len(self.y_col))]
        for i in range(len(self.y_col)):
            out_batches[i] = batches[self.y_col[out_names[i]]]

        shift = batches['shift'].tolist()
        flip = batches['flip'].tolist()
        low_vel = batches['low_vel'].tolist()

        X_batches = [[] for i in range(len(self.X_col))]
        #for i in range(len(self.X_col)):
        for i in range(1):
            X_batches[i] = np.asarray([self.__get_image_input(x, self.input_size, shift[idx], flip[idx], low_vel[idx], label=False)
                                       for idx,x in enumerate(in_batches[i])])
        for i in range(len(self.X_col)-1):
            X_batches[i + 1] = np.asarray([self.__get_vel_input(x) for idx,x in enumerate(in_batches[i+1])])

        y_batches = [[] for i in range(len(self.y_col))]
        for i in range(len(self.y_col)):
            y_batches[i] = np.array([self.__get_image_input(y, self.input_size, shift[idx], flip[idx], low_vel[idx], label=True)
            for idx,y in enumerate(out_batches[i])])

        return tuple(X_batches), tuple(y_batches)

    def __getitem__(self, index):
        batches = self.df[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

def custom_l2plusl1_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference)

    absolute_difference = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(absolute_difference)
    return mse + mae

def custom_arctan_loss(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    arc_tan_difference = tf.math.atan(absolute_difference)
    marce = tf.reduce_mean(arc_tan_difference)
    return marce

def custom_arctan_loss2(y_true, y_pred):
    atan_true = tf.math.atan(y_true)
    atan_pred = tf.math.atan(y_pred)
    absolute_difference = tf.abs(atan_true - atan_pred)
    marce = tf.reduce_mean(absolute_difference)
    return marce

def custom_mimusmean_loss(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true)
    y_true_minmean = y_true - true_mean
    y_pred_minmean = y_pred - true_mean
    mmae = tf.reduce_mean(tf.abs(y_true_minmean - y_pred_minmean))
    return mmae

def custom_MAAPE(y_true, y_pred):
    epsilon = 1*10**-7
    #AAPE = K.switch(K.equal(y_true, 0), tf.math.atan(tf.abs((y_true-y_pred)/(y_pred+epsilon))), tf.math.atan(tf.abs((y_true-y_pred)/y_true)))
    AAPE = tf.math.atan(tf.abs((y_true-y_pred)/(y_pred+epsilon)))
    MAAPE = tf.reduce_mean(AAPE)
    return MAAPE

class WindowGenerator():
    """
    WindowGenerator class copied from https://www.tensorflow.org/tutorials/structured_data/time_series
    creates data windows to train the time evolution networks
    Note: editted to return one dataset which is later split into train, validation, test
    """
    def __init__(self, input_width, label_width, shift,
               df, batchsize, label_columns=None, shuffle=True):
        self.shuffle = shuffle
        # Store the raw data.
        self.df = df
        self.bs = batchsize

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=self.shuffle,
          batch_size=self.bs)

        ds = ds.map(self.split_window)

        return ds

    @property
    def ds(self):
        return self.make_dataset(self.df)


    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the dataset
            result = next(iter(self.ds))
            # And cache it for next time
            self._example = result
        return result


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2, test=False, shuffle=True):
    """
    For more info, see https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
    """
    assert (train_split + val_split) == 1

    shuffle_size = ds_size
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    if not test:
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = None
    else:
        train_ds = None
        val_ds = None
        test_ds = ds

    return train_ds, val_ds, test_ds


def load_config(config_name):
    # load directories and hyperparameters from config file to config dict
    CONFIG_PATH = "."
    with open(os.path.join(CONFIG_PATH,config_name)) as file:
        config_dict = yaml.safe_load(file)
        return config_dict

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format='NHWC'):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape


def convert_im_to_256int(image):
    # ToDo: discuss using cutoff -> normalisation is bad idea
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 1:
                image[i][j] = 1
            elif image[i][j] < 0:
                image[i][j] = 0

    image = image * 255
    image = image.astype(np.uint8)
    return image

def find_boundary(image, giff = False):
    """
    Function to find the boundary of the channel
    :param image: on individual, grayscale, image of a velocity field
    :return contour[0]: the boundary of the channel, represented as numpy array of (x,y) coordinates
    """
    # First convert image such that boundary can be found
    image_copy = image.copy()
    if not giff:
        image_copy = convert_im_to_256int(image_copy)

    # Create binary image and find boundary, need 245 for image
    ret, thresh = cv2.threshold(image_copy, 245, 255, cv2.THRESH_BINARY)
    thresh = ~thresh

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    # Next part is only necessary for boundary visualisation and visual check images
    # cv2.imshow('image grayscale', image_copy)
    # cv2.imshow('Binary image', thresh)
    # image_color = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
    # image_color_copy = image_color.copy()
    # cv2.drawContours(image=image_color_copy, contours=contours, contourIdx=-1, color=(0, 255, 0),
    #                  thickness=1, lineType=cv2.LINE_AA)
    # cv2.imshow("contours", image_color_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return contours[0], thresh

def find_walls_inlet_outlet(image, scenario):
    boundary_coords, _ = find_boundary(image)
    boundary_coords = boundary_coords[:, 0]
    boundary_coords = [tuple(e) for e in boundary_coords]
    # Change order from x-y coordinate to row - column format
    boundary_coords = [(sub[1], sub[0]) for sub in boundary_coords]
    rows = [loc[0] for loc in boundary_coords]
    columns = [loc[1] for loc in boundary_coords]

    most_common_row = mode(rows)
    most_common_column = mode(columns)

    try:
        while True:
            rows.remove(most_common_row)
            columns.remove(most_common_column)
    except ValueError:
        pass

    sec_most_common_row = mode(rows)
    sec_most_common_column = mode(columns)

    if scenario != 'bend':
        inlet = np.amin(np.array([most_common_row, sec_most_common_row]))
        outlet = [np.amax(np.array([most_common_row, sec_most_common_row]))]
        # print(f"top row: {top}")
        # print(f"bottom row: {bottom}")
    else:
        inlet = most_common_row
        outlet = [185,186,184,183]
        # print('scenario = bend')
        # print(f"inlet row: {inlet}")
        # print(f"outlet rows: {outlet}")

    inlet_coords = []
    outlet_coords = []
    side_coords = []
    # Find most occurring row, smallest = inlet and biggest is outlet
    # ToDo: NOT TRUE for bend, there it is most occurring y coordinate, and split on range x coordinates
    for idx, coord in enumerate(boundary_coords):
        if coord[0] == inlet:
            # For straight channel we need to adjust the boundary
            if scenario =='straight':
                list_c = list(coord)
                list_c[0] = 5
                coord = tuple(list_c)
            inlet_coords.append(coord)
        elif coord[0] in outlet:
            if scenario =='bend' and coord[1] > 174:
                outlet_coords.append(boundary_coords[idx])
            elif scenario !='bend':
                outlet_coords.append(boundary_coords[idx])
        elif coord[1] == 196 and scenario =='straight':
            #For straight channel we need to adjust the boundary
            list_c = list(coord)
            list_c[1] = 195
            coord = tuple(list_c)
            side_coords.append(coord)
        else:
            side_coords.append(coord)

    # print(f"inlet coords: {inlet_coords}")
    # print(f"outlet coords: {outlet_coords}")
    # print(f"side coords: {side_coords}")

    #inlet_coords = [(loc[0]+1,loc[1]),(loc[0]-1, loc[1]) for loc in inlet_coords]

    return side_coords, inlet_coords, outlet_coords

def convert_pixel_to_vel(image):

    gray = cm.get_cmap('gray')
    colormap_lookup = gray(range(256))[:, 0]

    # ToDo: discuss using cutoff -> normalisation is bad idea
    image = convert_im_to_256int(image)

    converted_im = np.ones_like(image, dtype='float')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            converted_im[i][j] = colormap_lookup[image[i][j]]*0.5

    return converted_im

def make_gif(folder_path, image_folder, mask=None):
    frames = [None]*300
    print(image_folder)
    # image_list = [image for image in sorted(glob.glob(f"{folder_path}/{image_folder}/bifurcation/*.png"))]
    # true_im = cv2.imread(f'../DataGeneration/Data_Generated/Grayscale/TestSets/{image_folder}/bifurcation/bifurcation_w10.0_o3.25_v4.0_vp7_20000_vp7.png', cv2.IMREAD_GRAYSCALE)
    # true_im = cv2.resize(true_im, [256, 192])
    # contour, mask = find_boundary(true_im, giff = True)
    condition = np.stack((mask,) * 3, axis=-1) > 0.5
    mask_inv = ~mask
    for i in range(100):
        image = f"{folder_path}/{image_folder}/new_bifurcation_v_predicted_beat_{i}.png"
        #image = image_list[i+1]

        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, [256,192])
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        output_image = np.where(condition, image, np.stack((mask_inv,)*3,axis=-1))

        frames[i] = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow(f"image{idx}", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if idx == 0:
        #     continue

        #frames[i] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    imageio.mimsave(f"animation/{image_folder.split('/')[0]}.gif", frames, format='GIF', duration=0.05)


def make_gif_diff(folder_path):
    print(folder_path.split('/')[-1])

    frames = [None]*100
    for i in range(100):
        image = f"{folder_path}/bifurcation_diff_{i}.png"
        image = cv2.imread(image)
        frames[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageio.mimsave(f"animation/{folder_path.split('/')[-1]}_difference.gif", frames, format='GIF', duration=0.05)



# folder_path = '../DataGeneration/Data_Generated/Grayscale/TestSets'
# #folder_path = 'predicted_beats/bifurcation'
# image_folder = 'bifurcation_w10.0_o3.25_v4.0_real_3'
# make_gif(folder_path, image_folder)
# folder_path = 'predicted_beats'
# folder = 'bend'

# # for folder in os.listdir(folder_path):
# #     if folder == '.DS_Store' or folder == 'oldmodel_vp0':
# #         continue
# make_gif(folder_path, folder)

# plt.rcParams['font.size'] = '18'
# fig, ax = plt.subplots(figsize=(2, 10))
# fig.subplots_adjust(right=0.3)
#
# cmap = matplotlib.cm.jet
# norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
#
# fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=ax, orientation='vertical', label='velocity [m/s]')
# plt.tight_layout()
# fig.savefig(f"colorbar_velocity_0.5.jpeg")

