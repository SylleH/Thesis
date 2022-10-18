import pandas as pd

from trainer import *
from model import create_hyp_ETSD
from utils.ops import *
from utils.tools import *
import sys
import tensorflow as tf
from datetime import date
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('agg')

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
          data_dir = str(argv[2])
          history, checkpoint_filepath = trainer_AE(config, strategy, data_dir, total=True)

      elif argv[1] == "load":
          data_dir = argv[2]
          history, checkpoint_filepath = trainer_loaded_model(config, strategy, data_dir)
          print(checkpoint_filepath)
      else:
         print('Choose "AE", "TS", "total" or "load" as second argument :)')

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
          scenario = 'straight'
          checkpoint_filepath_total = str(argv[2]) + '/checkpoint'
          model_name = str(argv[3])
          fig_name = 'tot' +str(date.today())
          total_network = load_model('TS', checkpoint_filepath_total, date=model_name)

          test_generator, _ = create_input_multi_output_gen('data/NN_testset/', 192, 256, batch_size=96, previous_ts=1,
                                                             predicted_ts =5, test=True)
          # images, labels = test_generator[0]
          # label1, label2, label3, label4, label5 = labels\
          #     #,label6, label7, label8, label9, label10= labels
          # predicted1, predicted2, predicted3, predicted4, predicted5 = total_network.predict(test_generator) \
          #     #, predicted6, predicted7, predicted8, predicted9, predicted10 = total_network.predict(test_generator)
          #
          # # errors_0 = ErrorMetrics(images=test_generator, timeseries=None, prediction=None)
          # errors_1 = ErrorMetrics(images=label1, prediction=predicted1, scenario=scenario)
          # errors_2 = ErrorMetrics(images=label2, prediction=predicted2, scenario=scenario)
          # errors_3 = ErrorMetrics(images=label3, prediction=predicted3, scenario=scenario)
          # errors_4 = ErrorMetrics(images=label4, prediction=predicted4, scenario=scenario)
          # errors_5 = ErrorMetrics(images=label5, prediction=predicted5, scenario=scenario)
          # # errors_6 = ErrorMetrics(images=label6, prediction=predicted6, scenario=scenario)
          # # errors_7 = ErrorMetrics(images=label7, prediction=predicted7, scenario=scenario)
          # # errors_8 = ErrorMetrics(images=label8, prediction=predicted8, scenario=scenario)
          # # errors_9 = ErrorMetrics(images=label9, prediction=predicted9, scenario=scenario)
          # # errors_10 = ErrorMetrics(images=label10, prediction=predicted10, scenario=scenario)
          # error_list = [errors_1, errors_2, errors_3, errors_4, errors_5] \
          #               #,errors_6, errors_7, errors_8, errors_9, errors_10]
          #
          # df = pd.DataFrame(index=['boundary MAE rel','domain MAE rel','boundary ME rel', 'domain ME rel','domain RRMSE','boundary MAE abs','domain MAE abs',
          #                           'boundary ME abs','domain ME abs','domain RMSE',
          #                            'net flow error'],
          #                   columns=['pred1', 'pred2', 'pred3', 'pred4', 'pred5'])#, 'pred6', 'pred7', 'pred8', 'pred9', 'pred10'])
          #
          #
          # for idx, error in enumerate(error_list):
          #     column_name = 'pred' + str(idx+1)
          #
          #     avg_slip_mean_rel, avg_slip_mean_abs, max_slip_max_rel, max_slip_max_abs, avg_RMSE, avg_RRMSE, \
          #     avg_MAE_rel, avg_MAE_abs, max_ME_rel, max_ME_abs, df_max_abs, df_max_rel = error.evaluate()
          #     df.at['boundary MAE rel', column_name] = avg_slip_mean_rel
          #     df.at['boundary MAE abs', column_name] = avg_slip_mean_abs
          #     df.at['boundary ME rel', column_name] = max_slip_max_rel
          #     df.at['boundary ME abs', column_name] = max_slip_max_abs
          #     df.at['domain RMSE', column_name] = avg_RMSE
          #     df.at['domain RRMSE', column_name] = avg_RRMSE
          #     df.at['domain MAE rel', column_name] = avg_MAE_rel
          #     df.at['domain MAE abs', column_name] = avg_MAE_abs
          #     df.at['domain ME rel', column_name] = max_ME_rel
          #     df.at['domain ME abs', column_name] = max_ME_abs
          #
          #     net_flow_im = error.conservation(pred=False, ts=1)
          #     net_flow_pred = error.conservation(pred=True, ts=1)
          #     print(f"netflow image: {net_flow_im}")
          #     print(f"netflow prediction: {net_flow_pred}")
          #     net_flow_error = np.abs(net_flow_im - net_flow_pred)
          #     print(f"netflow error: {net_flow_error}")
          #     df.at['net flow error', column_name] = net_flow_error
          #
          #     # #ToDo: seaborn scatter plots (rel and abs) with name of prediction
          #     # df_max_abs['y_loc'] = -1*df_max_abs['y_loc']
          #     # abs = sns.relplot(data=df_max_abs, x='x_loc', y='y_loc', hue='group', size='count')
          #     # abs.set(ylim=(-192,0), xlim=(0,256))
          #     # abs.fig.suptitle(column_name +' max absolute error')
          #     # abs.savefig(column_name +'_max absolute error.png')
          #     #
          #     # df_max_rel['y_loc'] = -1 * df_max_rel['y_loc']
          #     # rel = sns.relplot(data=df_max_rel, x='x_loc', y='y_loc', hue='group', size='count')
          #     # rel.set(ylim=(-192, 0), xlim=(0, 256))
          #     # rel.fig.suptitle(column_name + ' max relative error')
          #     # rel.savefig(column_name + '_max relative error.png')
          #
          #     print(f"{column_name} done")
          #
          # print(df)
          # df.to_csv('results_csv/'+str(argv[2])+'_oldmodel_straight_GRU.csv')
          predict_one_beat(total_network, test_generator, t=1, scenario=scenario, directory ="oldmodel_straight_NN" )
          #test_scores = predict_and_plot_total(total_network, test_generator, predicted_ts=5)



      else:
         print('Choose "AE", "TS", "total_com" or "total_sep" as second argument :)')

   elif argv[0] == "hyptun":
       # check for GPUs and set strategy according
       print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
       if tf.config.list_physical_devices('GPU'):
           strategy = tf.distribute.MirroredStrategy()
       else:  # Use the Default Strategy = no distribution strategy (CPU training)
           strategy = tf.distribute.get_strategy()

       if argv[1] == "total":
           train_generator, val_generator = create_input_multi_output_gen('data/TS_seperately/', 192, 256,
                                                                          previous_ts=1,
                                                                          predicted_ts=5,
                                                                          batch_size=18, val_split=0.9)
           tuner = kt.Hyperband(create_hyp_ETSD,
                           objective='val_loss',
                           max_epochs=100,
                           factor=3,
                            distribution_strategy= strategy,
                           directory='hypertest',
                           project_name='Hyper_Total')

           es_callback = EarlyStopping(patience=20)

           tuner.search(train_generator, validation_data = val_generator,
                        epochs=250, callbacks=[es_callback])
           best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

           print(f"""
           The hyperparameter search is complete. The optimal parameters are {best_hps}.
           """)

           final_model = tuner.hypermodel.build(best_hps)
           modcheck_callback = ModelCheckpoint(filepath=os.path.join('model/hypertuned_total/', 'model_' + str(date.today())),
                                               save_weights_only=False, monitor='val_loss', mode='min',
                                               save_best_only=True)

           callbacks = [es_callback, modcheck_callback]

           with strategy.scope():
               history = final_model.fit(train_generator, validation_data=val_generator, epochs=250, callbacks=callbacks)

           plt.figure()
           plt.plot(history.history['loss'])
           plt.plot(history.history['val_loss'])
           plt.title('model loss')
           plt.ylabel('loss')
           plt.xlabel('epoch')
           plt.legend(['train', 'test'], loc='upper left')
           plt.savefig('hypermodel_trained_loss.png')
   else:
      print('Choose "train", "eval" or "hyptun" as first argument :)')


if __name__ == "__main__":
   # argv = ['train', 'TS', 'AE_straight_OVERFIT_val001_E126_f16_z150']
   # main(argv)
   main(sys.argv[1:])



