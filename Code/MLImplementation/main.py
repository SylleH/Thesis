from trainer import *
from utils.tools import *
import sys
import tensorflow as tf
from datetime import date


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
         _, encoder, _ = load_model("AE", checkpoint_filepath_AE, date='2022-06-20')
         history, checkpoint_filepath = trainer_TS(config, strategy, encoder)

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

         test_scores = predict_and_evaluate(autoencoder, test_generator, fig_name=fig_name)
         print(test_scores)

      elif argv[1] == "TS":
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'
         checkpoint_filepath_TS = str(argv[3]) + '/checkpoint/'

         _,encoder,_ = load_model('AE', checkpoint_filepath_AE, date='2022-06-20')
         TS_network = load_model('TS', checkpoint_filepath_TS)
         batch_size = 10 #BIGGEST NUMBER POSSIBLE = ALL TEST EXAMPLES (only influences inference speed)

         #Note: given dataset must be a sequence!
         _, _, _, test_ds = create_TS_dataset('data/NN_testset',encoder, train_split=1, val_split = 0, test=True)
         test_performance['NN'] = TS_network.evaluate(test_ds, verbose=1)


      elif argv[1] == "total":
         checkpoint_filepath_AE = str(argv[2]) + '/checkpoint/'
         checkpoint_filepath_TS = str(argv[3]) + '/checkpoint/'
         fig_name = 'E_TS_D_' + str(date.today())

         _, encoder, decoder = load_model('AE', checkpoint_filepath_AE, date='2022-06-20')
         TS_Network = load_model('TS', checkpoint_filepath_TS)
         _, _,_,test_ds = create_TS_dataset('data/NN_testset',encoder, train_split=1, val_split = 0, test=True)
         predicted = TS_Network.predict(test_ds)

         #ToDo: make different function for predicting and evaluating, because batches contain arrays, match with original images
         test_scores = predict_and_evaluate(decoder, predicted, fig_name=fig_name)
         print('test scored decoder: '+ test_scores)

      else:
         print('Choose "AE", "TS" or "total" as second argument :)')

   else:
      print('Choose "train" or "eval" as first argument :)')


if __name__ == "__main__":
   main(sys.argv[1:])



