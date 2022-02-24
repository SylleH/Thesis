from preprocessing import create_datagenerators, create_AE
import os
import yaml
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

#folder to load config file
CONFIG_PATH = "."

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH,config_name)) as file:
        config_dict = yaml.safe_load(file)
        return config_dict

#Load directories and hyperparameters
config = load_config("config.yaml")

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
model_path = model_dir+"AE_BS"+str(batch_size)+"_E"+str(epochs)
log_path = model_dir+"logs"
checkpoint_filepath = model_dir+"checkpoint/"

#define callbacks
tb_callback = TensorBoard(log_dir = log_path)
es_callback = EarlyStopping(patience=3)
modcheck_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'weight.{epoch:02d}-{val_loss:.2f}.h5'),
                                    save_weights_only=True, monitor= 'val_loss', mode='min',save_best_only=True)
callbacks = [tb_callback, es_callback, modcheck_callback]

train_generator, val_generator = create_datagenerators(data_dir,batch_size,img_height,img_width)
img_inputs = keras.Input(shape=(img_height, img_width, channels)) #shape of each sample needs to be supplied, this is after cropping
autoencoder = create_AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)



history = autoencoder.fit(train_generator, validation_data = val_generator,
                          batch_size = batch_size, epochs= epochs,
                          callbacks = callbacks)
autoencoder.save(model_path)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#test_scores = model.evaluate(test_ds)
