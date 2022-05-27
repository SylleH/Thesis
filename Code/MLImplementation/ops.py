"""
Extra helper functions
Author: Sylle Hoogeveen
"""

import os
from PIL import Image

scenario = "straight" #bifurcation, bend90 or branch
data_in_dir ="../DataGeneration/Data_generated/"+scenario+"/"
data_out_dir = "data/"+scenario+"/"

def custom_save(filepath, *args, **kwargs):
    """ Overwrite save function to save encoder and decoder separately during checkpoints """
    global autoencoder, encoder, decoder

    # fix name
    path, ext = os.path.splitext(filepath)

    # save encoder/decoder separately
    autoencoder.save(path +'_autoencoder', *args, **kwargs)
    encoder.save(path + '_encoder', *args, **kwargs)
    decoder.save(path + '_decoder', *args, **kwargs)

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

#create_cropped_images()
