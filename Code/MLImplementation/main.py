import pandas as pd

from trainer import *
from model import hyperbuild_TS_NN
from utils.ops import *
from utils.tools import *
import sys
import tensorflow as tf
from datetime import date
#import keras_tuner as kt


def main(argv):
   """
   Main function that can run train or evaluate scenario for either Autoencoder, TimeSeries
   or Total network.
   :param argv[0]: train or eval
   :param argv[1]: AE, TS or total
   :param argv[2]: AE model configuration name (not which specific model, that is specified by date)
   :param argv[3]: TS model configuration name (not which specific model, that is specified by date)
   :return:
   """
   config = load_config('config.yaml')

   if argv[0] == "train":
      # check for GPUs and set strategy according
      print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
      if tf.config.list_physical_devices('GPU'):
         strategy = tf.distribute.MirroredStrategy()
      else:  # Use the Default Strategy = no distribution strategy (CPU training)
         strategy = tf.distribute.get_strategy()

      if argv[1] == "AE":
         history, checkpoint_filepath = trainer_AE(config, strategy)

      elif argv[1] == "TS":
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'
         _, encoder, _ = load_model("AE", checkpoint_filepath_AE)
         history, checkpoint_filepath = trainer_TS(config, strategy, encoder)

      elif argv[1] == "total":
          history, checkpoint_filepath = trainer_AE(config, strategy, total=True)

      else:
         print('Choose "AE" or "TS" as second argument :)')

   elif argv[0] == "eval":
      val_performance = {}
      test_performance = {}
      #ToDo: create csv_file were model performance with model and data charateristics is stored
      #     or use MLflow..?

      if argv[1] == "AE":
         checkpoint_filepath = str(argv[2]) + '/checkpoint/'
         fig_name = str(argv[2])+'_'+str(date.today())

         autoencoder, _, _ = load_model("AE",checkpoint_filepath, date='2022-06-20')
         batch_size = 36 #BIGGEST NUMBER POSSIBLE = ALL TEST EXAMPLES (only influences inference speed)
         test_generator = create_test_datagen('data/AE_testset', batch_size, 192, 256)

         test_scores = predict_and_evaluate(autoencoder, test_generator, fig_name=fig_name,pipe=str(argv[1]))
         print(test_scores)

      elif argv[1] == "TS":
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'
         checkpoint_filepath_TS = str(argv[3]) + '/checkpoint/'

         _,encoder,_ = load_model('AE', checkpoint_filepath_AE, date='2022-06-20')
         TS_network = load_model('TS', checkpoint_filepath_TS, date='2022-06-28_tv_val001_noBN')
         batch_size = 10 #BIGGEST NUMBER POSSIBLE = ALL TEST EXAMPLES (only influences inference speed)

         #Note: given dataset must be a sequence!
         _, _, _, test_ds, _, _ = create_TS_dataset('data/NN_testset', encoder, input=50, shift=1, label=1,
                                                    train_split=1, val_split = 0, test=True)
         print(test_ds)
         test_performance['NN'] = TS_network.evaluate(test_ds, verbose=1)


      elif argv[1] == "total_sep":
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'
         checkpoint_filepath_TS = str(argv[3]) + '/checkpoint/'
         fig_name = 'E_TS_D_' + str(date.today())

         _, encoder, decoder = load_model('AE', checkpoint_filepath_AE, date='2022-06-20')
         TS_Network = load_model('TS', checkpoint_filepath_TS, date='2022-06-30_tvp_val010_noBN_w50')
         _, _,_,test_ds, test_generator, encoded = create_TS_dataset('data/NN_testset',encoder, input=50, shift=1, label=1,
                                                    batch_size = 20, train_split=1, val_split = 0, test=True)
         encoded_df = pd.DataFrame(encoded)

         for i in range(150):
            investigate_df = encoded_df.copy()
            investigate_df.iloc[:,i]=investigate_df.iloc[:,i]*100
            test_scores = predict_and_evaluate(decoder, investigate_df, fig_name=fig_name,pipe=str(argv[1]), images=test_generator,
                                               encoded=encoded, num=i)

         # TS_Network.evaluate(test_ds, verbose=1)
         # predicted = TS_Network.predict(test_ds)
         # pred_df = pd.DataFrame(predicted)
         # print(pred_df)
         # predicted,_ = np.split(predicted, indices_or_sections=[150], axis=1)

      elif argv[1] == "total_com":
          checkpoint_filepath_total = str(argv[2]) + '/checkpoint'
          fig_name = 'tot' +str(date.today())
          total_network = load_model('TS', checkpoint_filepath_total, date='2022-07-06')
          input_data, label_data = create_input_label_data('data/NN_testset/', 192, 256, test=True)
          test_scores = predict_and_evaluate(total_network, input_data, fig_name=fig_name, pipe=str(argv[1]), images=label_data)



      else:
         print('Choose "AE", "TS", "total_com" or "total_sep" as second argument :)')

   elif argv[0] == "hyptun":
      #ToDo: Not working yet :-(
      if argv[1] == "AE":
         train_generator, val_generator = create_train_val_datagen(data_dir='data/NN_testset',
                                                                   batch_size =36,
                                                                   img_height=192,
                                                                   img_width=256)

         tuner = kt.Hyperband(create_AE,
                           objective='val_accuracy',
                           max_epochs=10,
                           factor=3,
                           directory='hypertest',
                           project_name='AEtest')
         tuner.search(train_generator, validation_data = val_generator,
                          batch_size =36, epochs= 50)

      elif argv[1] == 'TS':
         data_dir = 'data/TS_seperately'
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'

         _, encoder, decoder = load_model('AE', checkpoint_filepath_AE, date='2022-06-20')
         window, train_ds, val_ds, test_ds, _, _ = create_TS_dataset(data_dir, encoder, batch_size=10,
                                                                     train_split=0.9, val_split=0.1, test=False)
         print(window)
         tuner = kt.Hyperband(hyperbuild_TS_NN,
                              objective='val_accuracy',
                              max_epochs=10,
                              factor=3,
                              directory='hypertest',
                              project_name='TStest')
         tuner.search(train_ds, validation_data=val_ds, batch_size=10, epochs=50)

         best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


   else:
      print('Choose "train" or "eval" as first argument :)')


if __name__ == "__main__":
   # argv = ['train', 'TS', 'AE_straight_OVERFIT_val001_E126_f16_z150']
   # main(argv)
   main(sys.argv[1:])



