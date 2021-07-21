###########################################################################################
# Libraries
###########################################################################################
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
from pathlib import Path

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from tensorflow import keras
from keras.utils import np_utils
import tensorflow as tf

from models import denseOutput
from utilities import evalBinaryClassifierCNN


wd = os.getcwd()

###########################################################################################
# Run model
###########################################################################################


def main():

    # Random seed
    tf.random.set_seed(1)

    # Path and folders
    path    = 'yourPath'
    outputs = [path + 'output_TDA/NewInnerFull_OptHP']
    loadTDA = [path + 'output_TDA/NewInnerFull_OptHP']
    suffix  = ['suffix_dim0',
              'suffix_dim1',
              'suffix_dim2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', nargs='+', type=int)
    parser.add_argument('--patch', nargs='+', type=int)
    args            = parser.parse_args()
    cv              = [args.cv[0]]
    runs            = [0,1,2]
    patches         = [args.patch[0]]
    normalize       = False # encoding: false, dist: true
    tda_dims        = [0,1,2]
    folderIDs       = [0]
    ensembleModel   = 'Dense'
    useForPredictio = 'encoding'
    model_nameTDA   = 'PI_CNN_model_None_dense_patch'
    partition_name  = '0CN_1AD_pat_1GO23_ML4HC_'

    # Loop over all patches
    for patch in patches:

        patchName   = patch

        print('CV: ',str(cv))
        print('run: ', str(runs))

        for folderID in folderIDs:
            output = outputs[folderID] + '/figures'
            Path(output).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_distance')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_eval')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_hdf5')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_losses')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_overviews')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'run_parameters')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(outputs[folderID], 'figures')).mkdir(parents=True, exist_ok=True)

            loadFeaturesFrom_TDA = loadTDA[folderID]
            for c in range(0,len(cv)):

                print('Working on fold '+str(c))

                # Load labels
                partition_suffix = partition_name + str(cv[c])
                partition_file   = path + 'partitions/partition_' + partition_suffix + '.npy'
                label_file       = path + 'partitions/labels_' + partition_suffix + '.npy'
                partition        = np.load(partition_file, allow_pickle='TRUE').item()
                labels           = np.load(label_file, allow_pickle='TRUE').item()


                # Get labels
                y_trn_in = [labels[p] for p in partition['train']]
                y_val_in = [labels[p] for p in partition['validation']]
                y_tst_in = [labels[p] for p in partition['test']]

                for r in runs:

                    # Load TDA features
                    for j in range(0,len(tda_dims)):

                        if useForPredictio == 'encoding':

                            filepath_enc_trn = os.path.join(os.path.join(loadFeaturesFrom_TDA, 'run_hdf5'),
                                                            'encoding_trn_'+model_nameTDA + str(patchName) +
                                                             '_run' + str(r) + 'CV_' + str(cv[c]) + '_tdadim[' +
                                                             str(tda_dims[j]) + ']'+suffix[j]+'.npy')
                            filepath_enc_val = os.path.join(os.path.join(loadFeaturesFrom_TDA, 'run_hdf5'),
                                                            'encoding_val_'+model_nameTDA  + str(patchName) +
                                                             '_run' + str(r) + 'CV_' + str(cv[c]) + '_tdadim[' +
                                                             str(tda_dims[j]) + ']'+suffix[j]+'.npy')
                            filepath_enc_tst = os.path.join(os.path.join(loadFeaturesFrom_TDA, 'run_hdf5'),
                                                            'encoding_tst_'+model_nameTDA  + str(patchName) +
                                                            '_run' + str(r) + 'CV_' + str(cv[c]) + '_tdadim[' +
                                                            str(tda_dims[j]) + ']'+suffix[j]+'.npy')
                            with open(filepath_enc_trn, 'rb') as f:
                                X_trn_TDA_dim = np.load(f, allow_pickle=True)
                            with open(filepath_enc_val, 'rb') as f:
                                X_val_TDA_dim = np.load(f, allow_pickle=True)
                            with open(filepath_enc_tst, 'rb') as f:
                                X_tst_TDA_dim = np.load(f, allow_pickle=True)

                            if j == 0:
                                X_trn_TDA = X_trn_TDA_dim
                                X_val_TDA = X_val_TDA_dim
                                X_tst_TDA = X_tst_TDA_dim
                            else:
                                X_trn_TDA = np.concatenate((X_trn_TDA, X_trn_TDA_dim), axis=1)
                                X_val_TDA = np.concatenate((X_val_TDA, X_val_TDA_dim), axis=1)
                                X_tst_TDA = np.concatenate((X_tst_TDA, X_tst_TDA_dim), axis=1)

                        else:
                            raise('Invalid choice for Preditio!')



                    # Create TDA Dataframe:
                    X_trn_df_TDA = pd.DataFrame(X_trn_TDA)
                    X_val_df_TDA = pd.DataFrame(X_val_TDA)
                    X_tst_df_TDA = pd.DataFrame(X_tst_TDA)


                    # Normalize
                    if normalize:
                        scalerTDA = preprocessing.StandardScaler().fit(X_trn_df_TDA)
                        X_trn_df_TDA_norm = pd.DataFrame(scalerTDA.transform(X_trn_df_TDA))
                        X_val_df_TDA_norm = pd.DataFrame(scalerTDA.transform(X_val_df_TDA))
                        X_tst_df_TDA_norm = pd.DataFrame(scalerTDA.transform(X_tst_df_TDA))



                        # Run dense etc on TDA
                        y_valTDA, y_pred_classTDA, accTDA, auc_valTDA, apsTDA, recallTDA,F1TDA,precisionTDA, tnTDA,\
                        tpTDA,fpTDA,fnTDA \
                                = run_model(X_trn_df_TDA_norm, y_trn_in, X_val_df_TDA_norm, y_val_in,
                                            X_tst_df_TDA_norm, y_tst_in,ensembleModel,output, cv[c],r,'TDA')
                    else:

                        # Run dense etc on TDA
                        y_valTDA, accTDA, auc_valTDA, apsTDA, recallTDA, F1TDA, precisionTDA,tnTDA, tpTDA, fpTDA,\
                        fnTDA,dist_trn, dist_val, y_pred_trn, y_pred_val, thresh_opt_trn,tn_tst, fp_tst, fn_tst,\
                        tp_tst, acc_tst, precision_tst,recall_tst, auc_tst, aps_tst, dist_tst, y_pred_tst\
                            = run_model(X_trn_df_TDA, y_trn_in, X_val_df_TDA, y_val_in,X_tst_df_TDA, y_tst_in,
                                        ensembleModel, output,cv[c], r, 'TDA')




                    # Create data frame
                    data_dfTDA = {'runID': str(runs),
                               'model': ensembleModel,
                               'n_epochs': 0,
                               'tn': tnTDA,
                               'fp': fpTDA,
                               'fn': fnTDA,
                               'tp': tpTDA,
                               'acc': accTDA,
                               'precision': precisionTDA,
                               'recall': recallTDA,
                               'auc': auc_valTDA,
                               'aps': apsTDA,
                               }
                    dfTDA = pd.DataFrame(data=data_dfTDA, index = [cv[c]])
                    dfTDA.to_csv(os.path.join(outputs[folderID] + '/run_overviews', ensembleModel +'TDA'+str(patch)+ '_run'
                                              + str(r) + 'CV_' + str(cv[c]) + '_results_overview.csv'))
                    data_dfTDA_tst = {'runID': str(runs),
                               'model': ensembleModel,
                               'n_epochs': 0,
                               'tn': tn_tst,
                               'fp': fp_tst,
                               'fn': fn_tst,
                               'tp': tp_tst,
                               'acc': acc_tst,
                               'precision': precision_tst,
                               'recall': recall_tst,
                               'auc': auc_tst,
                               'aps': aps_tst,
                               }
                    dfTDA_tst = pd.DataFrame(data=data_dfTDA_tst, index = [cv[c]])
                    dfTDA_tst.to_csv(os.path.join(outputs[folderID] + '/run_overviews', ensembleModel +'TDA'+str(patch)+ '_run'
                                              + str(r) + 'CV_' + str(cv[c]) + '_results_overview_tst.csv'))

                    # Distance:
                    dic_dist_trn = {'y_pred': y_pred_trn, 'dist': dist_trn, 'thresh': thresh_opt_trn}
                    df_dist_trn  = pd.DataFrame(dic_dist_trn)
                    df_dist_trn.to_csv(os.path.join(outputs[folderID] + '/run_distance', ensembleModel +
                                                    'TDA'+str(patch)+ '_run'+ str(r) + 'CV_' + str(cv[c]) +
                                                    '_results_dist_trn.csv'))

                    dic_dist_val = {'y_pred': y_pred_val, 'dist': dist_val, 'thresh': thresh_opt_trn}
                    df_dist_val  = pd.DataFrame(dic_dist_val)
                    df_dist_val.to_csv(os.path.join(outputs[folderID] + '/run_distance', ensembleModel +
                                                    'TDA'+str(patch)+ '_run'
                                              + str(r) + 'CV_' + str(cv[c]) + '_results_dist.csv'))

                    dic_dist_tst = {'y_pred': y_pred_tst, 'dist': dist_tst, 'thresh': thresh_opt_trn}
                    df_dist_tst  = pd.DataFrame(dic_dist_tst)
                    df_dist_tst.to_csv(os.path.join(outputs[folderID] + '/run_distance', ensembleModel +
                                                    'TDA' + str(patch) + '_run'
                                     + str(r) + 'CV_' + str(cv[c]) + '_results_dist_tst.csv'))



    return 0

def run_model(X_train, y_train, X_test, y_test,X_tst, y_tst, clf_choice, mydir, cv,r,name_suf):


    ####Get classifier:
    print('Getting classifier')

    if clf_choice == 'Dense':

        acc = 0
        aps = 0
        l1 = [0,0.1,0.001,0.0001,0.00001]
        shape = X_test.values.shape[1]
        for l1_i in l1:

            model = denseOutput(l1_i,shape)

            # compile model
            filepath = os.path.join(os.path.join(mydir,'Dense', 'run_hdf5'), 'Dense' +
                                    '_' + str(r) + 'CV_' + str(cv) + '_l1'+str(l1_i)+
                                    name_suf+'_nnet_run_test.hdf5')
            os.makedirs(os.path.join(mydir,'Dense', 'run_hdf5'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_losses'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_overviews'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_distance'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_eval'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_pats'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'run_parameters'), exist_ok=True)
            os.makedirs(os.path.join(mydir, 'Dense', 'figures'), exist_ok=True)

            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=0.0001),
                          metrics=None)
            model.summary()


            # train
            check = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                                       mode='auto')
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                                             mode='auto')

            try:
                y_train.shape
                y_trn = y_train
            except:
                y_trn = np_utils.to_categorical(y_train, 2)
            try:
                y_test.shape
                y_val = y_test
            except:
                y_val = np_utils.to_categorical(y_test, 2)

            try:
                y_tst.shape
            except:
                y_tst = np_utils.to_categorical(y_tst, 2)

            val_data   = (X_test.values, y_val)
            val_data_X = X_test.values
            tst_data_X = X_tst.values

            history = model.fit(x=X_train.values, y=y_trn,
                                batch_size=20,
                                epochs=500,
                                verbose=1,
                                callbacks=[check, earlyStopping],
                                validation_data=val_data,
                                )

            # Load the best weights
            model.load_weights(filepath)
            name = 'Dense' + '_run' + str(r) + 'CV_'+ str(cv)+ '_l1'+str(l1_i) +name_suf


            ####
            # Evaluate training and validation performance
            tn_out_trn, fp_out_trn, fn_out_trn, tp_out_trn, \
            acc_out_trn, precision_out_trn, recall_out_trn, \
            roc_auc_out_trn, aps_out_trn, dist_out_trn, \
            meanDist_out_trn, stdDist_out_trn, thresh_out_opt_trn, y_pred_out_trn \
            = evalBinaryClassifierCNN(model, X_train.values, y_trn, os.path.join(mydir,'Dense', 'run_eval'),
                                        labels=['Positives','Negatives'],
                                        plot_title='Evaluation train of ' + clf_choice,
                                        doPlots=True,
                                        batchsize=20,
                                        filename='train_'+name)

            tn_val_out, fp_val_out, fn_val_out, tp_val_out, \
            acc_val_out, precision_val_out, recall_val_out, \
            roc_auc_val_out, aps_val_out, dist_val_out, meanDist_val_out, stdDist_val_out,\
            thresh_opt, y_pred_val_out \
                = evalBinaryClassifierCNN(model, val_data_X, y_val, os.path.join(mydir,'Dense', 'run_eval'),
                                        labels=['Positives','Negatives'],
                                        plot_title='Evaluation validation of ' + clf_choice,
                                        doPlots=True,
                                        batchsize=20,
                                        filename='val_'+name,
                                        thresh_opt = thresh_out_opt_trn)



            # Clear keras
            tf.keras.backend.clear_session()

            if acc_val_out+aps_val_out>acc+aps:

                print('L1 '+ str(l1_i) + ' superior')
                tn = tn_val_out
                fp = fp_val_out
                fn = fn_val_out
                tp = tp_val_out
                acc = acc_val_out
                precision = precision_val_out
                recall = recall_val_out
                auc_val = roc_auc_val_out
                aps = aps_val_out
                F1 = 2 * (precision * recall) / (precision + recall)

                dist_trn       = dist_out_trn
                dist_val       = dist_val_out
                y_pred_trn     = y_pred_out_trn
                y_pred_val     = y_pred_val_out
                thresh_opt_trn = thresh_out_opt_trn

                tn_tst_out, fp_tst_out, fn_tst_out, tp_tst_out, acc_tst_out, precision_tst_out, \
                recall_tst_out, roc_auc_tst_out, \
                aps_tst_out, dist_tst_out, meanDist_tst_out, stdDist_tst_out, \
                thresh_opt, y_pred_tst_out \
                    = evalBinaryClassifierCNN(model, tst_data_X, y_tst, os.path.join(mydir, 'Dense', 'run_eval'),
                                              labels=[
                                                  'Positives',
                                                  'Negatives'],
                                              plot_title='Evaluation Test of ' + clf_choice,
                                              doPlots=True,
                                              batchsize=20,
                                              filename='test_' + name,
                                              thresh_opt=thresh_opt_trn)



    return y_val, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn,dist_trn,dist_val,y_pred_trn, \
           y_pred_val, thresh_opt_trn,tn_tst_out, fp_tst_out, fn_tst_out, tp_tst_out, acc_tst_out, \
           precision_tst_out,recall_tst_out, roc_auc_tst_out, aps_tst_out, dist_tst_out, y_pred_tst_out



if __name__ in "__main__":
    main()





