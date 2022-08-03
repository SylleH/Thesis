"""
Funtions needed at some point in other python files

Author: Sylle Hoogeveen
"""

import os
from PIL import Image
import tensorflow as tf
import numpy as np
import yaml
import cv2
import matplotlib
from matplotlib import cm

scenario = "straight" #bifurcation, bend90 or branch
data_in_dir ="../DataGeneration/Data_generated/"+scenario+"/"
data_out_dir = "data/"+scenario+"/"

class CustomDataGen(tf.keras.utils.Sequence):
    """
    Custom Image data generator for multiple inputs and outputs, adapted from:
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    """

    def __init__(self, df, X_col, y_col, batch_size, input_size, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_image_input(self, path, target_size):

        image = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale")
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

        return image_arr / 255.

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        in_batches = [[] for i in range(len(self.X_col))]
        in_names = [f"in{x}" for x in range(len(self.X_col))]
        for i in range(len(self.X_col)):
            in_batches[i] = batches[self.X_col[in_names[i]]]

        out_batches = [[] for i in range(len(self.y_col))]
        out_names = [f"out{x}" for x in range(len(self.y_col))]
        for i in range(len(self.y_col)):
            out_batches[i] = batches[self.y_col[out_names[i]]]

        X_batches = [[] for i in range(len(self.X_col))]
        for i in range(len(self.X_col)):
            X_batches[i] = np.asarray([self.__get_image_input(x, self.input_size) for x in in_batches[i]])

        y_batches = [[] for i in range(len(self.y_col))]
        for i in range(len(self.y_col)):
            y_batches[i] = np.array([self.__get_image_input(y, self.input_size) for y in out_batches[i]])

        return X_batches[0], tuple(y_batches)

    def __getitem__(self, index):
        batches = self.df[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

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

# def load_images(data, file_list, path, width, height):
#     for myFile in file_list:
#         myFile = path + myFile
#         image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
#         image = cv2.resize(image, (width, height))
#         data.append(image)
#     return data

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

def create_cropped_images():
    for image in os.listdir(data_in_dir):
        file, ext = os.path.splitext(image)
        im = Image.open(data_in_dir+image)
        cropped_im = im.crop((0,9,515,378)) #left, upper, right, lower
        cropped_im.save(data_out_dir+file+"_cropped.png")

#this can be used as preprocessing_function in ImageDataGenerator, but dimension issues
def crop(image):
    print(image.shape)
    start_y = 48
    start_x = 397
    cropped_image = image[start_y:(952 - start_y),start_x:(1578 - start_x), :]
    print(cropped_image.shape)
    return cropped_image

def find_boundary(image):
    """
    Function to find the boundary of the channel
    :param image: on individual, grayscale, image of a velocity field
    :return contour[0]: the boundary of the channel, represented as numpy array of (x,y) coordinates
    """
    # First convert image such that boundary can be found
    image_copy = image.copy()
    image_copy = (image_copy - np.min(image_copy))/np.ptp(image_copy)
    image_copy = image_copy*255
    image_copy = image_copy.astype(np.uint8)

    # Create binary image and find boundary
    ret, thresh = cv2.threshold(image_copy, 150, 255, cv2.THRESH_BINARY)
    thresh = ~thresh

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    # Next part is only necessary for boundary visualisation and visual check images
    # cv2.imshow('image grayscale',image_copy)
    # cv2.imshow('Binary image', thresh)
    # image_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # image_color_copy = image_color.copy()
    # cv2.drawContours(image=image_color_copy, contours=contours, contourIdx=-1, color=(0, 255, 0),
    #                  thickness=1, lineType=cv2.LINE_AA)
    # cv2.imshow("contours", image_color_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return contours[0], thresh

def find_walls_inlet_outlet(image):
    boundary_coords, _ = find_boundary(image)
    boundary_coords = boundary_coords[:, 0]
    boundary_coords = [tuple(e) for e in boundary_coords]
    # Change order from x-y coordinate to row - column format
    boundary_coords = [(sub[1], sub[0]) for sub in boundary_coords]

    # Use known inlet at row 4 and outlet at row 186 to differentiate between sides, inlet & outlet
    #ToDo: check if this is true for every image, also for branch/bifurcation?
    inlet_coords = []
    outlet_coords = []
    side_coords = []
    for idx, coord in enumerate(boundary_coords):
        if coord[0] == 4:
            inlet_coords.append(boundary_coords[idx])
        elif coord[0] == 186:
            outlet_coords.append(boundary_coords[idx])
        else:
            side_coords.append(boundary_coords[idx])

    # side_boundary = np.where(np.array(boundary_values) == 0)[0]
    # inout_boundary = np.where(np.array(boundary_values) != 0)[0]
    # side_coords = [boundary_coords[idx] for idx in side_boundary.tolist()]
    # inout_coords = [boundary_coords[idx] for idx in inout_boundary.tolist()]

    return side_coords, inlet_coords, outlet_coords

def convert_pixel_to_vel(image, name):
    # ToDo: discuss if normalization is legal or cheating, otherwise cut off...?
    gray = cm.get_cmap('gray')
    colormap_lookup = gray(range(256))[:, 0]

    image = (image - np.min(image)) / np.ptp(image)
    image = image * 255
    image = image.astype(np.uint8)

    converted_im = np.ones_like(image, dtype='float')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            converted_im[i][j] = colormap_lookup[image[i][j]]*0.5

    return converted_im