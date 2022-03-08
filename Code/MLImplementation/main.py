from preprocessing import create_datagenerators, create_AE
import os
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

#folder to load config file
CONFIG_PATH = "."

#check for GPUs and set strategy according
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
  strategy = tf.distribute.MirroredStrategy()
else:  # Use the Default Strategy = no distribution strategy (CPU training)
  strategy = tf.distribute.get_strategy()

#load directories and hyperparameters
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH,config_name)) as file:
        config_dict = yaml.safe_load(file)
        return config_dict

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
es_callback = EarlyStopping(patience=10)
modcheck_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'weight.{epoch:02d}-{val_loss:.2f}.h5'),
                                    save_weights_only=True, monitor= 'val_loss', mode='min',save_best_only=True)
if scenario == "test":
    callbacks = [es_callback]
else:
    callbacks = [tb_callback, es_callback, modcheck_callback]

#create data generators for training
train_generator, val_generator = create_datagenerators(data_dir,batch_size,img_height,img_width)

#create model with determined strategy
with strategy.scope():
    img_inputs = keras.Input(shape=(img_height, img_width, channels)) #shape of each sample needs to be supplied, this is after cropping
    autoencoder = create_AE(img_inputs, filters=filters, z_num=z_num,
                             repeat=repeat, num_conv=num_conv, conv_k=conv_k, last_k=last_k)

    history = autoencoder.fit(train_generator, validation_data = train_generator,
                          batch_size = batch_size, epochs= epochs, callbacks = callbacks)

predicted = autoencoder.predict(train_generator)
img_batch, _ = train_generator.next()
first_image = img_batch[0]
plt.figure()
plt.imshow(first_image)
plt.show()

img_batch = predicted
first_image = img_batch[0]
plt.figure()
plt.imshow(first_image)
plt.savefig(f"f{filters}_z{z_num}_nconv{num_conv}_r{repeat}_e{epochs}_jet.png")
plt.show()

#autoencoder.save(model_path)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f"loss_f{filters}_z{z_num}_nconv{num_conv}_r{repeat}_e{epochs}_jet.png")
plt.show()
#test_scores = model.evaluate(test_ds)
