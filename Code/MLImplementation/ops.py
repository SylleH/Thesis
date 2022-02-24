import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image

scenario = "straight" #bifurcation, bend90 or branch
data_dir = "data/"+scenario+"/"

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
    for filename in glob.glob("*.png"):
        file, ext = os.path.splitext(filename)
        im = Image.open(filename)
        cropped_im = im.crop((397,48,1181,904))
        cropped_im.save(data_dir+file+"_cropped.png")

#this can be used as preprocessing_function in ImageDataGenerator, but dimension issues
def crop(image):
    print(image.shape)
    start_y = 48
    start_x = 397
    cropped_image = image[start_y:(952 - start_y),start_x:(1578 - start_x), :]
    print(cropped_image.shape)
    return cropped_image