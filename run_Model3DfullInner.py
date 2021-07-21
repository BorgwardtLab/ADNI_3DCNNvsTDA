###########################################################################################
# Libraries
###########################################################################################
import pandas as pd
import numpy as np
import os
import time
import pickle
from pathlib import Path
from numpy.random import seed

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Model
from sklearn.utils import class_weight

from models import TDA_3DCNN_combinedModel
from utilities import evalBinaryClassifierCNN_datagen

########################################################################################################################
# Global parameters
########################################################################################################################

wd   = os.getcwd()
path = wd # Give path!

########################################################################################################################
class DataGenerator(keras.utils.Sequence):
    # Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 20200630

    def __init__(self, list_IDs, labels, batch_size=10, dim=(120,144,120), n_channels=1,
                 n_classes=2, shuffle=True, datadir = '/data'):

        'Initialization'
        self.dim        = dim
        self.batch_size = batch_size
        self.labels     = labels
        self.list_IDs   = list(list_IDs)
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.on_epoch_end()
        self.datadir    = datadir


    def on_epoch_end(self):

      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):

          # load sample - get the patch
          X[i,] = np.expand_dims(np.load(os.path.join(self.datadir, ID + '.npy'),allow_pickle=True),axis=3)

          # Store class
          y[i] = self.labels[ID]

      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'

      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]


      # Find list of IDs
      try:
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
      except:
        print(indexes)

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y

    def __getY__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        y = self.__data_generationY(list_IDs_temp)

        return y

    def __data_generationY(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)


      # Initialization
      y = np.empty((self.batch_size), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):

          # Store class
          y[i] = self.labels[ID]

      return keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getall__(self):
      'Generate one batch of data'

      # Find list of IDs
      list_IDs_temp = self.list_IDs

      # Generate data
      X, y = self.__data_generation_all(list_IDs_temp)

      return X, y

    def __data_generation_all(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)


      # Initialization
      X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
      y = np.empty((len(list_IDs_temp)), dtype=int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):

          # Store sample
          X[i,] = np.expand_dims(np.load(os.path.join(self.datadir, ID + '.npy')),axis=3)

          # Store class
          y[i] = self.labels[ID]

      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation_Y(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Initialization
        y = np.empty((len(list_IDs_temp)), dtype=int)
        print(len(list_IDs_temp))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store class
            y[i] = self.labels[ID]

        return keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getall_Y__(self):
      'Generate one batch of data'

      # Find list of IDs
      list_IDs_temp = self.list_IDs

      # Generate data
      y = self.__data_generation_Y(list_IDs_temp)

      return y


if __name__ in "__main__":
    ###########################################################################################
    # Input section
    ###########################################################################################

    # Run info general selection
    clf              = 'dense'
    clf_3D           = 'gap'
    choice_modelTDA  = 'None'
    choice_model3D   = 'CNNflexi'
    partition_suffix = '0CN_1AD_pat_1GO23_ML4HC_'


    # Data folders and dimensions
    data_folder_3D = path+'DATA/3D/FullIM'

    # 3D image dimensions
    dim0p    = 120
    dim1p    = 144
    dim2p    = 120
    patchdim = (dim0p, dim1p, dim2p)
    n_channels_3D = 1

    # TDA - No in use here!
    dim0tda = 50
    dim1tda = 50
    dim2tda = 0
    patchdimTDA = (dim0tda, dim1tda, dim2tda)
    n_channels_tda = 1
    useDimsTDA = [[0, 1, 2]]


    # Output folder
    outputtensorboard = path + 'fullInnner'
    Path(outputtensorboard).mkdir(parents=True, exist_ok=True)
    output = [path + '/output_3D/InnerFull']



    # Create folders if needed
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_distance')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_eval')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_hdf5')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_losses')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_overviews')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_parameters')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'figures')).mkdir(parents=True, exist_ok=True)
    print('Output: ' + output)

    #####################################
    #       Get hyperparameters         #
    #####################################

    # 3D Parameters
    file   = 'modelParameters/3DFullInnerBrain/None_CNNflexi_dense_patch217_run0CV_0_tdadim[0]_param3D.pkl'
    infile = open(file, 'rb')
    params3D_loaded = pickle.load(infile)
    infile.close()

    n_channels_3D = params3D_loaded['n_channels']
    patchdim      = params3D_loaded['patchdim']
    BNloc_3D      = params3D_loaded['BNloc']
    useDO_3D      = params3D_loaded['useDO']
    n_filter_1_3D = params3D_loaded['n_filters_1']
    n_filter_2_3D = params3D_loaded['n_filters_2']
    k_size_3D     = params3D_loaded['kernel_size']
    l2_3D         = params3D_loaded['l2']
    l1_den3D      = params3D_loaded['l1_den']
    l1_act3D      = params3D_loaded['l1_act']
    sample_shape_3D= params3D_loaded['sample_shape']
    clf_3D         = params3D_loaded['clf']
    n_layers_3D    = 5

    file   = 'modelParameters/3DFullInnerBrain/None_CNNflexi_dense_patch217_run0CV_0_tdadim[0]_paramCombi.pkl'
    infile = open(file, 'rb')
    inputCombi_loaded = pickle.load(infile)
    infile.close()

    clf       = inputCombi_loaded['clf']
    n_classes = inputCombi_loaded['n_classes']
    n_epochs  = 1500
    bs        = inputCombi_loaded['bs']
    lr        = inputCombi_loaded['lr']
    l1_tail   = inputCombi_loaded['l1']

    # Hyperparameters used for all model arm which are shared
    pat = 200
    input_Combi = {'clf': clf,
                   'n_classes': n_classes,
                   'n_epochs': n_epochs,
                   'bs': bs,
                   'lr': lr,
                   'pat': pat,
                   'l1': l1_tail}


    ###########################################################################################
    # Run model
    ###########################################################################################

    # Configure GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


    # Loop over different CV fold
    for cv in range(0,5):

        # Model choices
        modeluse = choice_modelTDA + '_' + choice_model3D + '_' + clf
        print('Model use: ' + modeluse)
        print('CV: ' + str(cv))
        NAME = modeluse+'_fullIM'+'_'+'{}'.format(int(time.time()))


        #####################################
        #              LOAD DATA            #
        #####################################

        # Get partition and labels containing information on training, validation, test samples and labels
        partition_file   = path + 'partitions/partition_' + partition_suffix + str(cv) + '.npy'
        label_file       = path + 'partitions/labels_' + partition_suffix+ str(cv) + '.npy'
        partition        = np.load(partition_file, allow_pickle='TRUE').item()  # IDs
        labels           = np.load(label_file, allow_pickle='TRUE').item()  # Labels


        # Prepare 3D image data loader
        print('Loading 3D MRI data')

        # Data set
        params_dataloader = {'dim': patchdim,
                      'batch_size': input_Combi['bs'],
                      'n_classes': n_classes,
                      'n_channels': 1,
                      'shuffle': True,
                      'datadir': data_folder_3D,
                      }
        params_dataloaderTest = {'dim': patchdim,
                             'batch_size': input_Combi['bs'],
                             'n_classes': n_classes,
                             'n_channels': 1,
                             'shuffle': False,
                             'datadir': data_folder_3D,
                             }
        training_generator   = DataGenerator(partition['train'], labels, **params_dataloader)
        validation_generator = DataGenerator(partition['validation']
                                             [0:np.floor(len(partition['validation']) / params_dataloaderTest['batch_size']).astype(
                                                 int) * params_dataloaderTest['batch_size']],labels, **params_dataloaderTest)
        test_generator = DataGenerator(partition['test'][0:np.floor(len(partition['test']) / params_dataloaderTest['batch_size']).astype(
                                                 int) * params_dataloaderTest['batch_size']],labels, **params_dataloaderTest)

        # Sample shapes
        sample_shape_3D = (input_Combi['bs'], dim0p, dim1p, dim2p, 1)

        # Hyperparameters specific for 3DCNN
        params_3D = {'n_channels': n_channels_3D,
                     'datadir': data_folder_3D,
                     'patchdim': patchdim,
                     'BNloc': BNloc_3D,
                     'useDO': useDO_3D,
                     'n_filters_1': n_filter_1_3D,
                     'n_filters_2': n_filter_2_3D,
                     'kernel_size': k_size_3D,
                     'l2': l2_3D,
                     'l1_den': l1_den3D,
                     'l1_act': l1_act3D,
                     'sample_shape': sample_shape_3D,
                     'clf': clf_3D,
                     'n_layers': n_layers_3D
                     }



        # load TDA data, sample size and hyperparameters - Not used here
        X_dicTDA = np.empty((1, 1, 1))
        y_dicTDA = 0
        sample_shape_TDA = 0
        params_TDA = {}


        for counter, runIDuse in enumerate([0,1,2]):

            # Random seeds for reproducible results
            seed(runIDuse)
            tf.random.set_seed(runIDuse)

            # Name
            name = modeluse + '_fullIM_run' + str(runIDuse) + 'CV_' + str(cv)

            #####################################
            #           build the model         #
            #####################################
            print(params_3D)

            # Model input
            input3D = {'sample_shape': sample_shape_3D,
                       'params': params_3D}
            inputTDA = {'sample_shape': sample_shape_TDA,
                        'params': params_TDA}

            # Get the model
            model = TDA_3DCNN_combinedModel(inputTDA, input3D, input_Combi, choice_modelTDA, choice_model3D)

            # Compile the model
            monitor_var     = 'loss'
            monitor_val_var = 'val_loss'
            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=input_Combi['lr']),
                          metrics=None)
            model.summary()


            #  Early stopping and check points
            filepath = os.path.join(os.path.join(output, 'run_hdf5'), name + '_nnet_run_test.hdf5')
            check = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = input_Combi['pat'],
                                                             mode='auto')
            tensorboard       = TensorBoard(log_dir = os.path.join(outputtensorboard,'logs/{}'.format(NAME)))
            callbacks         = [check, earlyStopping, tensorboard]

            # Class weights
            y_trn             = training_generator.__getall_Y__()
            class_weights     = class_weight.compute_class_weight('balanced', np.unique(y_trn[:, 1]), y_trn[:, 1])
            class_weight_dict = dict(enumerate(class_weights))


            # Actual training
            history = model.fit(training_generator,
                                validation_data=validation_generator,
                                batch_size=input_Combi['bs'],
                                epochs=n_epochs,
                                verbose=1,
                                callbacks=callbacks,
                                class_weight=class_weight_dict,
                                shuffle=True
                                )

            # Load the best weights
            model.load_weights(filepath)

            # Save model
            filepath_model = os.path.join(os.path.join(output, 'run_hdf5'), 'MODEL_' + name + '_nnet_run_test.h5')
            model.save(filepath_model)

            # Get encoding
            encoder = Model(model.input, model.layers[-2].output)
            encoder.summary()


            # Evaluate training data and save
            tn_trn, fp_trn, fn_trn, tp_trn, acc_trn, precision_trn, recall_trn, roc_auc_trn, aps_trn, dist_trn, \
            meanDist_trn, stdDist_trn, thresh_opt_trn, y_pred_trn = \
                evalBinaryClassifierCNN_datagen(model, training_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                        doPlots=False,
                                        batchsize=input_Combi['bs'],
                                        filename='trn' + name)
            dic_dist_trn = {'y_pred': y_pred_trn, 'dist': dist_trn, 'thresh': thresh_opt_trn}
            df_dist_trn = pd.DataFrame(dic_dist_trn)

            # Evaluate validation and test set and save
            tn, fp, fn, tp, acc, precision, recall, roc_auc, aps, dist, meanDist, stdDist, thresh_opt, y_pred = \
                evalBinaryClassifierCNN_datagen(model, validation_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                        doPlots=False,
                                        batchsize=input_Combi['bs'],
                                        filename='val_' + name,
                                        thresh_opt = thresh_opt_trn)
            dic_dist = {'y_pred': y_pred, 'dist': dist, 'thresh': thresh_opt}
            df_dist = pd.DataFrame(dic_dist)

            tn_tst, fp_tst, fn_tst, tp_tst, acc_tst, precision_tst, recall_tst, roc_auc_tst, aps_tst, dist_tst,\
            meanDist_tst, stdDist_tst, thresh_opt_tst, y_pred_tst = \
                evalBinaryClassifierCNN_datagen(model, test_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation Test of ' + modeluse + '_' + clf,
                                        doPlots=True,
                                        batchsize=input_Combi['bs'],
                                        filename='test_' + name,
                                        thresh_opt=thresh_opt_trn)
            dic_dist_tst = {'y_pred': y_pred_tst, 'dist': dist_tst, 'thresh': thresh_opt}
            df_dist_tst = pd.DataFrame(dic_dist_tst)



            # Save results to df
            data_df = {'runID': str(runIDuse),
                       'data_folder': data_folder_3D,
                       'data_folderTDA': 'nan',
                       'model': modeluse,
                       'clf': clf,
                       'n_epochs': n_epochs,
                       'tn': tn,
                       'fp': fp,
                       'fn': fn,
                       'tp': tp,
                       'acc': acc,
                       'precision': precision,
                       'recall': recall,
                       'auc': roc_auc,
                       'aps': aps,
                       'meanDist': meanDist,
                       'stdDist': stdDist
                       }
            df = pd.DataFrame(data=data_df, index=[runIDuse])
            df.to_csv(os.path.join(os.path.join(output, 'run_overviews'), name + '_results_overview.csv'))

            data_df_tst = {'runID': str(runIDuse),
                       'data_folder': data_folder_3D,
                       'data_folderTDA': 'nan',
                       'model': modeluse,
                       'clf': clf,
                       'n_epochs': n_epochs,
                       'tn': tn_tst,
                       'fp': fp_tst,
                       'fn': fn_tst,
                       'tp': tp_tst,
                       'acc': acc_tst,
                       'precision': precision_tst,
                       'recall': recall_tst,
                       'auc': roc_auc_tst,
                       'aps': aps_tst,
                       'meanDist': meanDist_tst,
                       'stdDist': stdDist_tst
                       }
            df_tst = pd.DataFrame(data=data_df_tst, index=[runIDuse])
            df_tst.to_csv(os.path.join(os.path.join(output, 'run_overviews'), name + '_results_overview_tst.csv'))


            # Distances
            df_dist_trn.to_csv(os.path.join(os.path.join(output, 'run_distance'), name + '_results_dist_trn.csv'))
            df_dist.to_csv(os.path.join(os.path.join(output, 'run_distance'), name + '_results_dist.csv'))
            df_dist_tst.to_csv(os.path.join(os.path.join(output, 'run_distance'), name + '_results_dist_tst.csv'))

            # Save parameters
            if choice_model3D != 'None':
                file = os.path.join(os.path.join(output, 'run_parameters'), name + '_param3D.pkl')
                with open(file, 'wb') as f:
                    pickle.dump(params_3D, f)

            if choice_modelTDA != 'None':
                file = os.path.join(os.path.join(output, 'run_parameters'), name + '_paramTDA.pkl')
                with open(file, 'wb') as f:
                    pickle.dump(params_TDA, f)


            file = os.path.join(os.path.join(output, 'run_parameters'), name + '_paramCombi.pkl')
            with open(file, 'wb') as f:
                pickle.dump(input_Combi, f)

            # Clear keras
            tf.keras.backend.clear_session()


