from utils.preprocessing import *
from utils.tools import *
import os
import yaml
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import date
import matplotlib.pyplot as plt

# folder to load config file
CONFIG_PATH = "."

# check for GPUs and set strategy according
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
else:  # Use the Default Strategy = no distribution strategy (CPU training)
    strategy = tf.distribute.get_strategy()

# load directories and hyperparameters
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH,config_name)) as file:
        config_dict = yaml.safe_load(file)
        return config_dict

config = load_config("config.yaml")
# expand to see all loaded configs
network = config["architecture"]
scenario = config["data"]["scenario"]
data_dir = "data/"+scenario+"/"
model_dir = config["data"]["model_dir"]

img_height = config["preprocess"]["img_height"]
img_width = config["preprocess"]["img_width"]
channels = config["preprocess"]["channels"]

filters = config["AE"]["filters"]
z_num = config["AE"]["z_num"]
num_conv= config["AE"]["num_conv"]
conv_k= config["AE"]["conv_k"]
last_k= config["AE"]["last_k"]
repeat= config["AE"]["repeat"]

batch_size = config["AE"]["training"]["batch_size"]
epochs = config["AE"]["training"]["epochs"]
model_path = model_dir+"AE_"+str(scenario)+"_BS"+str(batch_size)+"_E"+str(epochs)+"_f"+str(filters)+"_z"+str(z_num)
log_path = model_path+"/logs"
checkpoint_filepath = model_path+"/checkpoint/"
fig_name = f"loss_f{filters}_z{z_num}_nconv{num_conv}_r{repeat}_e{epochs}_"+str(date.today())


#define callbacks
tb_callback = TensorBoard(log_dir = log_path)
es_callback = EarlyStopping(patience=20)
modcheck_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'model_'+str(date.today())),
                                    save_weights_only=False, monitor= 'val_loss', mode='min',save_best_only=True)
if scenario == "test":
    callbacks = [es_callback]
else:
    callbacks = [tb_callback, es_callback, modcheck_callback]

#create data generators for training AE
# if network == 'AE':
#   train_generator, val_generator = create_train_val_datagen(data_dir,batch_size,img_height,img_width)
#
#
#   #create model and train with determined strategy
#   with strategy.scope():
#       img_inputs = keras.Input(shape=(img_height, img_width, channels)) #shape of each sample needs to be supplied, this is after cropping
#       autoencoder = create_AE(img_inputs, filters=filters, z_num=z_num,
#                              repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)
#
#       history = train_and_store(autoencoder, train_generator, val_generator,
#                           batch_size, epochs, callbacks, fig_name)


"""
RUN CODE ABOVE ON CLUSTER (TRAINING), RUN CODE BELOW ON LAPTOP (PREDICT & EVALUATE)
"""
#create generator for testing
test_generator = create_test_datagen('data/test/', img_height, img_width)

#make predictions and evaluate model
if os.path.exists(os.path.join(checkpoint_filepath, 'model_2022-05-26')): #+str(date.today()))):
   model = keras.models.load_model(os.path.join(checkpoint_filepath, 'model_2022-05-26'))#+str(date.today())))
   print(model.summary())
   #ToDo: change names of these layers such that the are the same for each model
   encoder = keras.Model(model.input, model.get_layer(name='12_fc').output)
   decoder = keras.Model(model.get_layer(name='12_fc').output, model.get_layer(name='9_deconv').output)



   #test_scores = predict_and_evaluate(model, test_generator, fig_name= f"f{filters}_z{z_num}_nconv{num_conv}_r{repeat}_e{epochs}"+str(date.today()))
   #print(test_scores)
else:
   print('train model first!')

#use encoded images to train NN
window = create_TS_dataset(encoder, test_generator)
print('Input shape:', window.example[0].shape)

encoded_inputs = keras.Input(shape=window.example[0].shape[1:])
#print(encoded_inputs)
# create_TS_network can take NN, RNN_LSTM and RNN_GRU as inputs for model
TS_network = create_TS_network(x=encoded_inputs, onum=100, model='RNN_GRU')
print(TS_network.summary())
history = train_and_store(TS_network, window.train ,window.val, batch_size=10, epochs=20, callbacks=[es_callback], fig_name='testNN')

#store performance of models
val_performance = {}
performance = {}
val_performance['NN'] = TS_network.evaluate(window.val)
performance['NN'] = TS_network.evaluate(window.test, verbose=0)
