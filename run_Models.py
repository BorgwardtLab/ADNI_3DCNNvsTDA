###########################################################################################
# Libraries
###########################################################################################
import pandas as pd
import os
from tensorflow import keras
from keras.utils import np_utils
import numpy as np
from keras.models import Model
import tensorflow as tf
import argparse

from sklearn.utils import class_weight
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
from pathlib import Path
from numpy.random import seed

from utilities import getArrayWithTest, create_split, evalBinaryClassifierCNN
from models import TDA_3DCNN_combinedModel


########################################################################################################################
# Global parameters
########################################################################################################################

wd   = os.getcwd()
path = 'your path'

########################################################################################################################

def getInnerPatches():
    ids = np.zeros((6, 6, 6))
    innerPatch = []
    for patch in range(0, 216):
        patchid2 = patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
        patchid1 = np.floor((patch - np.floor(patch / 36) * 36) / 6)
        patchid0 = np.floor(patch / 36)
        ids[int(patchid0), int(patchid1), int(patchid2)] = patch
        print('Patch ' + str(patch) + ': ' + str(patchid0) + ' ' + str(patchid1) + ' ' + str(patchid2))

        if patchid0 > 0 and patchid0 < 5 and patchid1 > 0 and patchid1 < 5 and patchid2 > 0 and patchid2 < 5:
            innerPatch.append(patch)

    return innerPatch


if __name__ in "__main__":

    ###########################################################################################
    # Input section
    ###########################################################################################

    # Run info general selection
    clf_3D    = 'gap'
    modelsTDA = ['None', 'PI_CNN_model', 'PI_CNN_model']
    models3D  = ['CNNflexi', 'None', 'CNNflexi']
    partition_suffix = '0CN_1AD_pat_1GO23_ML4HC_'
    allpatches = getInnerPatches()


    # Data folders - indicate the relevant folders containing data - for fullInner brain TDA use
    # the same model as for TDA patches and HC
    dataFoldersAll = [path+'Data3Dpatches',
                      'nan',
                      'nan',
                      path+'Data3DHC']

    data_folder_TDAs = ['nan',
                        path+'dataPatchesPIs',
                        path+'data_HCPIs',
                        'nan',
                        ]
    loadParametersFrom = ['modelParameters/3Dpatch',
                          'modelParameters/TDA',
                          'modelParameters/TDA',
                          'modelParameters/3DHC']


    # Output folder
    outputtensorboard = path + 'outputTB'
    Path(outputtensorboard).mkdir(parents=True, exist_ok=True)
    outputs = [path + 'output_3D/Patches',
               path + 'output_TDA/Patches',
               path + 'output_TDA/BrainFSU_HCmodel',
               path + 'output_3D/BrainFSU_HCmodel',
              ]

    dim3D   = [[30,36,30],
               [30,36,30],
               [30,36,30],
               [33,46,48]]

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', nargs='+', type=int, default=1)
    parser.add_argument('--modelind', nargs='+', type=int, default=1)
    parser.add_argument('--patch', nargs='+', type=int, default=91)
    args      = parser.parse_args()
    thisCV    = args.cv[0]
    patch_in  = args.patch[0]
    model_ind = args.modelind[0]
    clfind    = 0
    folderID  = args.folderID[0]

    # Option for patches of brain subunits
    if model_ind == 0:
        useTDAdimsHere = [0]
    else:
        useTDAdimsHere = [0,1,2]

    if folderID <2:
        thesepatches = allpatches
    elif folderID==2:
        thesepatches = [99,48] # left and right HC
    else:
        thesepatches = [patch_in]
    print('Working on patches '+str(thesepatches))


    for patch in thesepatches:
        for useDimsTDA_ID in useTDAdimsHere:
            for cv in [thisCV]:

                # Model choices
                clf             = 'dense'
                choice_modelTDA = modelsTDA[model_ind]
                choice_model3D  = models3D[model_ind]
                modeluse        = choice_modelTDA + '_' + choice_model3D + '_' + clf
                print('Model use: ' + modeluse)
                print('CV: ' + str(cv))
                NAME = modeluse+'_'+ str(patch)+'_'+'{}'.format(int(time.time()))
                if model_ind == 1:
                    loadParameters = True
                    useTensorBoard = False


                # Data folders
                output          = os.path.join(outputs[folderID],str(patch))
                data_folder_TDA = os.path.join(data_folder_TDAs[folderID],str(patch))
                data_folder_3D  = os.path.join(path, dataFoldersAll[folderID])

                # Create folders if needed
                Path(output).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_distance')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_eval')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_hdf5')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_losses')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_overviews')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'run_parameters')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(output, 'figures')).mkdir(parents=True, exist_ok=True)
                loadfrom = output
                print('Output: ' + output)


                # 3D image dimensions
                dim0p    = dim3D[folderID][0]
                dim1p    = dim3D[folderID][1]
                dim2p    = dim3D[folderID][2]#
                patchdim = (dim0p, dim1p, dim2p)
                n_channels_3D = 1

                # TDA
                dim0tda = 50
                dim1tda = 50
                dim2tda = 0
                patchdimTDA    = (dim0tda, dim1tda, dim2tda)
                n_channels_tda = 1
                useDimsTDA     = [useDimsTDA_ID]


                # load Parameters:
                n_layers_3D   = 2
                if model_ind!= 1:

                    # 3D Parameters
                    file   = loadParametersFrom[folderID]+'_param3D.pkl'
                    infile = open(file, 'rb')
                    params3D_loaded = pickle.load(infile)
                    infile.close()

                    n_channels_3D = params3D_loaded['n_channels']
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

                    file = loadParametersFrom[folderID]+'_paramCombi.pkl'
                    infile = open(file, 'rb')
                    inputCombi_loaded = pickle.load(infile)
                    infile.close()

                    clf = inputCombi_loaded['clf']
                    n_classes = inputCombi_loaded['n_classes']
                    n_epochs = inputCombi_loaded['n_epochs']
                    bs = inputCombi_loaded['bs']
                    lr = inputCombi_loaded['lr']
                    l1_tail = inputCombi_loaded['l1']
                else:

                    # TDA parameters
                    file    = os.path.join(loadParametersFrom[folderID]+ str(0) + ']_paramCombi.pkl')
                    fileTDA = os.path.join(loadParametersFrom[folderID]+ str(0) + ']_paramTDA.pkl')
                    infile  = open(file, 'rb')
                    param_tail = pickle.load(infile)
                    infile.close()
                    infile = open(fileTDA, 'rb')
                    params_TDA= pickle.load(infile)
                    infile.close()

                    clf       = param_tail['clf']
                    n_classes = param_tail['n_classes']
                    n_epochs  = param_tail['n_epochs']
                    bs        = param_tail['bs']
                    lr        = param_tail['lr']
                    l1_tail   = param_tail['l1']




                ###########################################################################################
                # Run model
                ###########################################################################################

                # Configure GPU
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

                #####################################
                #       Get hyperparameters         #
                #####################################

                # Hyperparameters used for all model arm which HAVE TO BE shared
                pat = 200
                input_Combi = {'clf': clf,
                               'n_classes': n_classes,
                               'n_epochs': n_epochs,
                               'bs': bs,
                               'lr': lr,
                               'pat': pat,
                               'l1': l1_tail}


                #####################################
                #              LOAD DATA            #
                #####################################

                # Get partition and labels containing information on training, validation, test samples and labels
                partition_file   = path + 'partitions/partition_' + partition_suffix + str(cv) + '.npy'
                label_file       = path + 'partitions/labels_' + partition_suffix+ str(cv) + '.npy'
                partition        = np.load(partition_file, allow_pickle='TRUE').item()  # IDs
                labels           = np.load(label_file, allow_pickle='TRUE').item()  # Labels

                #try:
                # Load 3D image data
                if choice_model3D != 'None':

                    print('Loading 3D MRI data')

                    # Get the specified patches
                    print('Working on patch ' + str(patch))
                    patchid2 = patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
                    patchid1 = np.floor((patch - np.floor(patch / 36) * 36) / 6)
                    patchid0 = np.floor(patch / 36)
                    patchid  = (patchid0, patchid1, patchid2)

                    # Get the actual array
                    X_trn_3D, y_trn_3D, X_val_3D, y_val_3D, X_tst_3D, y_tst_3D = getArrayWithTest(data_folder_3D, patch,
                                                                                                  partition, labels, patchdim,
                                                                                                n_channels_3D)


                    # Make categorical
                    y_trn = np_utils.to_categorical(y_trn_3D, n_classes)
                    y_val = np_utils.to_categorical(y_val_3D, n_classes)
                    y_tst = np_utils.to_categorical(y_tst_3D, n_classes)

                    # Sample shapes
                    sample_shape_3D = (input_Combi['bs'], dim0p, dim1p, dim2p, 1)

                    # hyperparameters specific for 3DCNN
                    params_3D = {'n_channels': n_channels_3D,
                                 'datadir': data_folder_3D,
                                 'patchid': patch,
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
                else:
                    sample_shape_3D = (1, 1, 1)
                    patchid2 = np.nan
                    patchid1 = np.nan
                    patchid0 = np.nan
                    patchid = (patchid0, patchid1, patchid2)
                    params_3D = {}


                # load TDA data, sample size and hyperparameters
                if choice_modelTDA == 'None':
                    X_dicTDA = np.empty((1, 1, 1))
                    y_dicTDA = 0
                    sample_shape_TDA = 0
                    params_TDA = {}
                else:
                    print('Loading persistence images')

                    # Get data
                    X_dicTDA, y_dicTDA = create_split(
                        data_dir=data_folder_TDA,
                        dimensions=useDimsTDA,
                        partition_file=partition_file,
                        labels_file=label_file,
                        reshape=True,
                        resolution=[dim0tda, dim1tda],
                        use_T1w=False)

                    # Labels
                    y_trnTDA = np.array(y_dicTDA['train'])
                    y_valTDA = np.array(y_dicTDA['validation'])
                    y_tstTDA = np.array(y_dicTDA['test'])
                    y_trnTDA = np_utils.to_categorical(y_trnTDA, n_classes)
                    y_valTDA = np_utils.to_categorical(y_valTDA, n_classes)
                    y_tstTDA = np_utils.to_categorical(y_tstTDA, n_classes)

                    # Sample shape
                    if choice_modelTDA == 'MLP':
                        sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1])
                    else:
                        if len(useDimsTDA) == 3:
                            sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1], np.asarray(X_dicTDA['train']).shape[2], 1)
                        else:
                            sample_shape_TDA = (np.asarray(X_dicTDA['train']).shape[1:])
                    print('Sample shape TDA: ', sample_shape_TDA)


                # Loop over all runs - option to skip
                runIDsuse = [0,1,2]
                for counter, runIDuse in enumerate(runIDsuse):

                    # Random seeds for reproducible results
                    seed(runIDuse)
                    tf.random.set_seed(runIDuse)

                    # Name
                    name = modeluse + '_patch' + str(patch) + '_run' + str(runIDuse) + 'CV_' + str(cv) +\
                           '_tdadim' + str(useDimsTDA)


                    #####################################
                    #           build the model         #
                    #####################################

                    # Model input
                    input3D = {'sample_shape': sample_shape_3D,
                               'params': params_3D}
                    inputTDA = {'sample_shape': sample_shape_TDA,
                                'params': params_TDA}
                    if choice_modelTDA == 'None':
                        input = X_trn_3D
                        val_data = (X_val_3D, y_val)
                        val_data_X = X_val_3D
                        tst_data = (X_tst_3D, y_tst)
                        tst_data_X = X_tst_3D
                    else:
                        if choice_model3D == 'None':

                            # One TDA dim:
                            if len(useDimsTDA) == 3:
                                allX_TDA = np.asarray(X_dicTDA['train'])
                                allXval_TDA = np.asarray(X_dicTDA['validation'])
                                input = [allX_TDA[:, :, :, 0], allX_TDA[:, :, :, 1], allX_TDA[:, :, :, 2]]
                                val_data = ([allXval_TDA[:, :, :, 0], allXval_TDA[:, :, :, 1], allXval_TDA[:, :, :, 2]], y_val)
                                val_data_X = [allXval_TDA[:, :, :, 0], allXval_TDA[:, :, :, 1], allXval_TDA[:, :, :, 2]]
                                tst_data = (
                                [allXtst_TDA[:, :, :, 0], allXtst_TDA[:, :, :, 1], allXtst_TDA[:, :, :, 2]], y_tst)
                                tst_data_X = [allXtst_TDA[:, :, :, 0], allXtst_TDA[:, :, :, 1], allXtst_TDA[:, :, :, 2]]
                            else:
                                input = np.asarray(X_dicTDA['train'])
                                val_data = (np.asarray(X_dicTDA['validation']), y_val)
                                val_data_X = np.asarray(X_dicTDA['validation'])
                                tst_data = (np.asarray(X_dicTDA['test']), y_tst)
                                tst_data_X = np.asarray(X_dicTDA['test'])
                        else:

                            # Multiple TDA dims simultanteously
                            if len(useDimsTDA) == 3:
                                allX_TDA = np.asarray(X_dicTDA['train'])
                                allXval_TDA = np.asarray(X_dicTDA['validation'])
                                allXtst_TDA = np.asarray(X_dicTDA['test'])
                                input = [allX_TDA[:, :, :, 0], allX_TDA[:, :, :, 1], allX_TDA[:, :, :, 2], X_trn_3D]
                                val_data = (
                                [allXval_TDA[:, :, :, 0], allXval_TDA[:, :, :, 1], allXval_TDA[:, :, :, 2], X_val_3D], y_val)
                                val_data_X = [allXval_TDA[:, :, :, 0], allXval_TDA[:, :, :, 1], allXval_TDA[:, :, :, 2], X_val_3D]
                                tst_data = (
                                    [allXtst_TDA[:, :, :, 0], allXtst_TDA[:, :, :, 1], allXtst_TDA[:, :, :, 2],
                                     X_tst_3D], y_tst)
                                tst_data_X = [allXtst_TDA[:, :, :, 0], allXtst_TDA[:, :, :, 1], allXtst_TDA[:, :, :, 2],
                                              X_tst_3D]
                            else:
                                input = [np.asarray(X_dicTDA['train']), X_trn_3D]
                                val_data = ([np.asarray(X_dicTDA['validation']), X_val_3D], y_val)
                                val_data_X = [np.asarray(X_dicTDA['validation']), X_val_3D]
                                tst_data = ([np.asarray(X_dicTDA['test']), X_tst_3D], y_tst)
                                tst_data_X = [np.asarray(X_dicTDA['test']), X_tst_3D]

                    # Get the model
                    model = TDA_3DCNN_combinedModel(inputTDA, input3D, input_Combi, choice_modelTDA, choice_model3D)

                    # Compile the model
                    monitor_var = 'loss'
                    monitor_val_var = 'val_loss'
                    model.compile(loss='binary_crossentropy',
                                  optimizer=keras.optimizers.Adam(lr=input_Combi['lr']),
                                  metrics=None)
                    model.summary()


                    #  Early stopping and check points
                    filepath      = os.path.join(os.path.join(output, 'run_hdf5'), name + '_nnet_run_test.hdf5')
                    check         = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                                       save_best_only=True, mode='auto')
                    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=input_Combi['pat'],
                                                                     mode='auto')
                    tensorboard = TensorBoard(log_dir=os.path.join(outputtensorboard, 'logs/{}'.format(NAME)))
                    callbacks = [check, earlyStopping, tensorboard]

                    # Class weights
                    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_trn[:, 1]), y_trn[:, 1])
                    class_weight_dict = dict(enumerate(class_weights))


                    # Actual training
                    history = model.fit(x=input, y=y_trn,
                                        batch_size=input_Combi['bs'],
                                        epochs=input_Combi['n_epochs'],
                                        verbose=1,
                                        callbacks=callbacks,
                                        validation_data=val_data,
                                        class_weight=class_weight_dict,
                                        shuffle=True)

                    # Load the best weights
                    model.load_weights(filepath)

                    # Save model
                    filepath_model = os.path.join(os.path.join(output, 'run_hdf5'), 'MODEL_'+name + '_nnet_run_test.h5')
                    model.save(filepath_model)

                    # Get encoding
                    encoder = Model(model.input, model.layers[-2].output)
                    encoder.summary()

                    # Predict
                    x_trn_encoded = encoder.predict(input)
                    x_val_encoded = encoder.predict(val_data_X)
                    x_tst_encoded = encoder.predict(tst_data_X)
                    weights = model.layers[-1].get_weights()

                    # Save
                    x_trn_encoded_flat = np.reshape(x_trn_encoded, (x_trn_encoded.shape[0], np.prod(x_trn_encoded.shape[1:])))
                    x_val_encoded_flat = np.reshape(x_val_encoded, (x_val_encoded.shape[0], np.prod(x_val_encoded.shape[1:])))
                    x_tst_encoded_flat = np.reshape(x_tst_encoded, (x_tst_encoded.shape[0], np.prod(x_tst_encoded.shape[1:])))
                    filepath_enc_trn = os.path.join(os.path.join(loadfrom, 'run_hdf5'), 'encoding_trn_' + name + '.npy')
                    filepath_enc_val = os.path.join(os.path.join(loadfrom, 'run_hdf5'), 'encoding_val_' + name + '.npy')
                    filepath_enc_tst = os.path.join(os.path.join(loadfrom, 'run_hdf5'), 'encoding_tst_' + name + '.npy')
                    filepath_enc_weights = os.path.join(os.path.join(loadfrom, 'run_hdf5'), 'encoding_weights_preClassLayer' + name + '.npy')
                    with open(filepath_enc_trn, 'wb') as f:
                        np.save(f, x_trn_encoded_flat, allow_pickle=True)
                    with open(filepath_enc_val, 'wb') as f:
                        np.save(f, x_val_encoded_flat, allow_pickle=True)
                    with open(filepath_enc_tst, 'wb') as f:
                        np.save(f, x_tst_encoded_flat, allow_pickle=True)
                    with open(filepath_enc_weights, 'wb') as f:
                        np.save(f, weights, allow_pickle=True)


                    # Evaluate training data and save
                    tn_trn, fp_trn, fn_trn, tp_trn, acc_trn, precision_trn, recall_trn, roc_auc_trn, aps_trn, dist_trn, \
                    meanDist_trn, stdDist_trn, thresh_opt_trn, y_pred_trn = \
                        evalBinaryClassifierCNN(model, input, y_trn, os.path.join(output, 'run_eval'),
                                                labels=['Negatives', 'Positives'],
                                                plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                                doPlots=False,
                                                batchsize=input_Combi['bs'],
                                                filename='trn' + name)
                    dic_dist_trn = {'y_pred': y_pred_trn, 'dist': dist_trn, 'thresh': thresh_opt_trn}
                    df_dist_trn = pd.DataFrame(dic_dist_trn)


                    # Evaluate validation and test set and save
                    tn, fp, fn, tp, acc, precision, recall, roc_auc, aps, dist, meanDist, stdDist, thresh_opt, y_pred = \
                        evalBinaryClassifierCNN(model, val_data_X, y_val, os.path.join(output, 'run_eval'),
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
                        evalBinaryClassifierCNN(model, tst_data_X, y_tst, os.path.join(output, 'run_eval'),
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
                               'data_folderTDA': data_folder_TDA,
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
                               'data_folderTDA': data_folder_TDA,
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

                    # Save run distance to decision boundary
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


