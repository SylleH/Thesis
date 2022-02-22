from model import AE
from preprocessing import create_datasets, preprocess
from tensorflow import keras

scenario = "straight" #bifurcation, bend90 or branch
data_dir = "data/"+scenario+"/"

batch_size = 32
img_height = 856 #after cropping, original = 952
img_width = 784 #after cropping, original = 1578

filters = 10
z_num = 100
num_conv= 1 #4
conv_k=3
last_k=3
repeat= 1 #0

train_generator, val_generator = create_datasets(data_dir,batch_size,img_height,img_width)


img_inputs = keras.Input(shape=(img_height, img_width,3)) #shape of each sample needs to be supplied, this is after cropping
z, out, autoencoder = AE(img_inputs, filters = filters, z_num = z_num, batch_size = batch_size, repeat=repeat, num_conv=num_conv,conv_k=conv_k, last_k=last_k)
autoencoder.compile(loss = "mse", optimizer="adam", metrics=["mse"])
print(autoencoder.summary())
#keras.utils.plot_model(autoencoder, "autoencoder_arc.png")

history = autoencoder.fit(train_generator, validation_data = val_generator, batch_size = batch_size, epochs=2) #fit without supplying y and validation_split
#test_scores = model.evaluate(test_ds)
#model.save("path_to_my_model")