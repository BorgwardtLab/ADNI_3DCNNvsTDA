###########################################################################################
# Libraries
###########################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from pathlib import Path



wd   = os.getcwd()


###########################################################################################
# Run model
###########################################################################################
def main():

    cv = [0,1,2,3,4]
    runs = [0,1,2]


    patches    = getInnerPatches()
    prefix     = 'allPatches'
    use3DorPIs = ['PI']  #Options: '3D, PI, 3DPI - combine either 3D, TDA or both predictions

    for use3DorPI in use3DorPIs:

        doEnsemblePrediction = True
        ensembleModel        = 'LogisticRegression'
        pi_dim               = 3 #options:0,1,2 or:3 (combine 0-2)
        name_patches         = 'inner'
        path                 = 'yourPath'
        prefixPI             = 'val_PI_CNN_model_None_dense_patch'
        suffix_PI            = ['_tdadim[', '_tdadim[','_tdadim[']
        name                 = 'None_CNNflexi_dense_patch' # 3D model name
        partition_name       = '0CN_1AD_pat_1GO23_ML4HC_'

        for c in range(0,len(cv)):

            # Get labels
            partition_suffix = partition_name + str(cv[c])

            # Get paths
            if use3DorPI == 'PI':
                output           = [path+'output_TDA/Patches']
                saveto           = path+'output_TDA/Patches/LR_Ensembles/Ensembel_innerPatchesTDA'

            elif use3DorPI == '3D':
                pi_dim = 0
                output = [path + 'output_3D/Patches']
                saveto = path + 'output_3D/Patches/LR_Ensembles/Ensembel_innerPatches3D'

            elif use3DorPI == '3DPI':
                outputTDA = [path + 'output_TDA/NewPatches']
                output3D  = [path + 'output_3D/Patches']
                saveto = path + 'output_TDA/NewPatches/LR_Ensembles/Ensembel_innerPatches3DTDA'
            else:
                raise('Invalid choice for use3DorPI')

            # Create folders if needed
            Path(saveto).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_distance')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_eval')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_hdf5')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_losses')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_overviews')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'run_parameters')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(saveto, 'figures')).mkdir(parents=True, exist_ok=True)


            # Configure GPU (optional)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


            # Load labels
            partition_file = path + 'partitions/partition_' + partition_suffix + '.npy'
            label_file     = path + 'partitions/labels_' + partition_suffix + '.npy'
            partition      = np.load(partition_file, allow_pickle='TRUE').item()
            labels         = np.load(label_file, allow_pickle='TRUE').item()
            pats           = np.array(partition['validation'])


            for r in range(0,len(runs)):

                print('Working on fold ' + str(cv[c]))
                print('Working on run ' + str(runs[r]))

                if use3DorPI == '3DPI':

                    # TDA features
                    if pi_dim == 3:

                        for pi_dim_use in [0, 1, 2]:

                            X_tst_dim, y_tstTDA = predictAllPatchesPerRun(partition, 'test', labels, outputTDA, name,
                                                                          [cv[c]], [r],patches,'PI',pi_dim_use, suffix_PI,
                                                                          prefixPI)
                            X_val_dim, y_valTDA = predictAllPatchesPerRun(partition, 'validation', labels, outputTDA, name,
                                                                          [cv[c]], [r],patches, 'PI',pi_dim_use, suffix_PI,
                                                                          prefixPI)
                            X_trn_dim, y_trnTDA = predictAllPatchesPerRun(partition, 'train', labels, outputTDA, name,
                                                                          [cv[c]],[runs[r]],patches, 'PI', pi_dim_use,suffix_PI, prefixPI)

                            if pi_dim_use == 0:
                                X_trnTDA = X_trn_dim[:, patches]
                                X_valTDA = X_val_dim[:, patches]
                                X_tstTDA = X_tst_dim[:, patches]
                            else:
                                X_trnTDA = np.concatenate((X_trnTDA, X_trn_dim[:, patches]), axis=1)
                                X_valTDA = np.concatenate((X_valTDA, X_val_dim[:, patches]), axis=1)
                                X_tstTDA = np.concatenate((X_tstTDA, X_tst_dim[:, patches]), axis=1)

                            # X_trn_df = pd.DataFrame(X_trn_all)  # sel_patches])#
                            # X_val_df = pd.DataFrame(X_val_all)
                            # X_tst_df = pd.DataFrame(X_tst_all)
                    else:
                        X_tstTDA, y_tstTDA = predictAllPatchesPerRun(partition, 'test', labels, output, name, [cv[c]],
                                                                     [r],patches, use3DorPI, pi_dim,suffix_PI, prefixPI)
                        X_valTDA, y_valTDA = predictAllPatchesPerRun(partition, 'validation', labels, output, name,
                                                                     [cv[c]], [r], patches,use3DorPI, pi_dim,
                                                                     suffix_PI, prefixPI)
                        X_trnTDA, y_trnTDA = predictAllPatchesPerRun(partition, 'train', labels, output, name, [cv[c]],
                                                                     [runs[r]],patches, use3DorPI, pi_dim, suffix_PI,
                                                                     prefixPI)

                        if doEnsemblePrediction:
                            bestacc_val = np.max(X_val[0, -216:])

                            # print('Best train acc: '+str(bestacc_trn))
                            print('Best validation acc: ' + str(bestacc_val))

                            X_trn_df = pd.DataFrame(X_trn[:, patches])  # sel_patches])#
                            X_val_df = pd.DataFrame(X_val[:, patches])
                            X_tst_df = pd.DataFrame(X_tst[:, patches])



                    # 3D features
                    X_tst3D, y_tst = predictAllPatchesPerRun(partition, 'test', labels, output3D, name, [cv[c]],
                                                           [r], patches, '3D', pi_dim, suffix_PI,prefixPI)
                    X_val3D, y_val = predictAllPatchesPerRun(partition, 'validation', labels, output3D, name, [cv[c]],
                                                           [r], patches, '3D', pi_dim, suffix_PI,prefixPI)
                    X_trn3D, y_trn = predictAllPatchesPerRun(partition, 'train', labels, output3D, name, [cv[c]],
                                                           [runs[r]], patches, '3D', pi_dim, suffix_PI,prefixPI)

                    if doEnsemblePrediction:

                        bestacc_val = np.max(X_val3D[0, -216:])

                        # print('Best train acc: '+str(bestacc_trn))
                        print('Best validation acc 3D: ' + str(bestacc_val))

                        if pi_dim == 3:
                            X_trn = np.concatenate((X_trnTDA, X_trn3D[:, patches]), axis=1)
                            X_val = np.concatenate((X_valTDA, X_val3D[:, patches]), axis=1)
                            X_tst = np.concatenate((X_tstTDA, X_tst3D[:, patches]), axis=1)
                        else:
                            X_trn = np.concatenate((X_trnTDA[:, patches], X_trn3D[:, patches]), axis=1) # sel_patches])#
                            X_val = np.concatenate((X_valTDA[:, patches], X_val3D[:, patches]), axis=1)
                            X_tst = np.concatenate((X_tstTDA[:, patches], X_tst3D[:, patches]), axis=1)

                        # Combine TDA and 3D features
                        X_trn_df = pd.DataFrame(X_trn)
                        X_val_df = pd.DataFrame(X_val)
                        X_tst_df = pd.DataFrame(X_tst)


                else:
                    if pi_dim==3:

                        for pi_dim_use in [0,1,2]:

                            X_tst_dim, y_tst = predictAllPatchesPerRun(partition, 'test', labels, output, name,
                                                                       [cv[c]], [r],patches, use3DorPI,pi_dim_use,
                                                                       suffix_PI, prefixPI)
                            X_val_dim, y_val = predictAllPatchesPerRun(partition, 'validation', labels, output, name, [cv[c]], [r],
                                                                       patches, use3DorPI,pi_dim_use,suffix_PI,prefixPI)
                            X_trn_dim, y_trn = predictAllPatchesPerRun(partition, 'train', labels, output, name, [cv[c]], [runs[r]],
                                                                       patches, use3DorPI, pi_dim_use,
                                                                       suffix_PI,prefixPI)



                            if pi_dim_use == 0:
                                X_trn_all = X_trn_dim[:, patches]
                                X_val_all = X_val_dim[:, patches]
                                X_tst_all = X_tst_dim[:, patches]
                            else:
                                X_trn_all = np.concatenate((X_trn_all, X_trn_dim[:, patches]), axis = 1)
                                X_val_all = np.concatenate((X_val_all, X_val_dim[:, patches]), axis = 1)
                                X_tst_all = np.concatenate((X_tst_all, X_tst_dim[:, patches]), axis=1)

                            X_trn_df = pd.DataFrame(X_trn_all)
                            X_val_df = pd.DataFrame(X_val_all)
                            X_tst_df = pd.DataFrame(X_tst_all)
                    else:
                        X_tst, y_tst = predictAllPatchesPerRun(partition, 'test', labels, output, name, [cv[c]], [r],
                                                               patches, use3DorPI, pi_dim,suffix_PI, prefixPI)
                        X_val, y_val = predictAllPatchesPerRun(partition, 'validation', labels, output, name, [cv[c]],
                                                               [r], patches, use3DorPI, pi_dim, suffix_PI,prefixPI)
                        X_trn, y_trn = predictAllPatchesPerRun(partition, 'train', labels, output, name, [cv[c]],
                                                               [runs[r]], patches, use3DorPI,pi_dim, suffix_PI,prefixPI)


                        if doEnsemblePrediction:
                            bestacc_val = np.max(X_val[0, -216:])

                            #print('Best train acc: '+str(bestacc_trn))
                            print('Best validation acc: ' + str(bestacc_val))

                            X_trn_df = pd.DataFrame(X_trn[:, patches])  # sel_patches])#
                            X_val_df = pd.DataFrame(X_val[:, patches])
                            X_tst_df = pd.DataFrame(X_tst[:, patches])


                if doEnsemblePrediction:

                    # Check for nans
                    if np.isnan(X_trn_df).any().any() or np.isnan(X_val_df).any().any():
                        nancols_trn = X_trn_df.columns[X_trn_df.isna().any()].tolist()
                        nancols_val = X_val_df.columns[X_val_df.isna().any()].tolist()
                        nancols = nancols_val+nancols_trn
                    else:
                        nancols = []

                    if (X_trn_df == 0).all().values.any():
                        zerocols_trn = list(np.where((X_trn_df == 0).all().values)[0])
                    else:
                        zerocols_trn = []

                    dropcols =   nancols+  zerocols_trn

                    X_trn_df = X_trn_df.drop(columns=dropcols)
                    X_val_df = X_val_df.drop(columns=dropcols)
                    X_tst_df = X_tst_df.drop(columns=dropcols)


                    # Run log. Regression
                    y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision, tn, tp,fp,fn, \
                    y_tst, y_pred_class_tst, acc_tst, auc_tst, aps_tst, recall_tst, F1_tst, precision_tst, \
                    tn_tst, tp_tst, fp_tst, fn_tst\
                        = run_model(X_trn_df, y_trn, X_val_df, y_val,X_tst_df, y_tst,
                                    ensembleModel,saveto, cv[c],runs[r])

                # Create data frame
                data_df = {'runID': str(r),
                           'model': ensembleModel,
                           'n_epochs': 0,
                           'tn': tn,
                           'fp': fp,
                           'fn': fn,
                           'tp': tp,
                           'acc': acc,
                           'precision': precision,
                           'recall': recall,
                           'auc': auc_val,
                           'aps': aps,
                           }
                df = pd.DataFrame(data=data_df, index=[r])
                if use3DorPI == 'PI':
                    df.to_csv(os.path.join(saveto,'run_overviews',ensembleModel +prefix+ '_' + use3DorPI +'['+str(pi_dim)
                                           + ']_run' + str(runs[r]) + 'CV_' + str(cv[c]) +
                                           name_patches+'_results_overview.csv'))
                else:
                    df.to_csv(os.path.join(saveto, 'run_overviews', ensembleModel +prefix+ '_' + use3DorPI+ '_run' + str(runs[r]) + 'CV_' +
                                           str(cv[c]) + '_results_overview.csv'))

                # Create data frame
                data_df_tst = {'runID': str(r),
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
                df_tst = pd.DataFrame(data=data_df_tst, index=[r])
                if use3DorPI == 'PI':
                    df_tst.to_csv(os.path.join(saveto, 'run_overviews', ensembleModel +prefix+ '_' + use3DorPI + '[' + str(pi_dim)
                                           + ']_run' + str(runs[r]) + 'CV_' + str(cv[c]) +
                                           name_patches + '_results_overview_tst.csv'))
                else:
                    df_tst.to_csv(os.path.join(saveto, 'run_overviews',
                                           ensembleModel+prefix + '_' + use3DorPI + '_run' + str(runs[r]) + 'CV_' +
                                           str(cv[c]) + '_results_overview_tst.csv'))

    return 0

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

def predictAllPatchesPerRun(partition, subset, labels, output, name, cv, runs, patches,
                            use3DorPI, pi_dim, suffix_PI, prefixPI):

    suffix = ''

    if len(cv) > 1:
        raise ('Only one fold at a time for now!')

    # initialize
    missingPatches = []

    if subset == 'test':
        tail_resdist = ']'+ suffix+'_results_dist_tst.csv'
        tail_resAcc  = ']'+ suffix+'_results_overview.csv'
    elif subset == 'validation':
        tail_resdist = ']'+ suffix+'_results_dist.csv'
        tail_resAcc  = ']'+ suffix+'_results_overview.csv'
    elif subset == 'train':
        tail_resdist = ']'+ suffix+'_results_dist_trn.csv'
        tail_resAcc  = ']'+ suffix+'_results_overview.csv'
    else:
        raise('Wrong subset!!')

    if subset == 'test':
        tail_resdist3D = ']_results_dist_tst.csv'
        tail_resAcc3D  = ']_results_overview.csv'
    elif subset == 'validation':
        tail_resdist3D = ']_results_dist.csv'
        tail_resAcc3D  = ']_results_overview.csv'
    elif subset == 'train':
        tail_resdist3D = ']_results_dist_trn.csv'
        tail_resAcc3D  = ']_results_overview.csv'
    else:
        raise('Wrong subset!!')

    # Loop over all validation patients:
    X = np.empty((len(partition[subset]), 216 * 3))

    # Loop through all csvs and get the relavant information
    map_Dist         = np.zeros((6, 6, 6, len(partition[subset])))
    map_stdDist      = np.zeros((6, 6, 6, len(partition[subset])))
    map_PredLabel    = np.zeros((6, 6, 6, len(partition[subset])))
    map_stdPredLabel = np.zeros((6, 6, 6, len(partition[subset])))
    map_ACC          = np.zeros((6, 6, 6))
    map_stdACC       = np.zeros((6, 6, 6))

    # Load data for all runs and CV and average:
    mean_absdists = np.empty((216, len(partition[subset]), len(cv)))
    std_absdists  = np.empty((216, len(partition[subset]), len(cv)))
    mean_dists    = np.empty((216, len(partition[subset]), len(cv)))
    std_dists     = np.empty((216, len(partition[subset]), len(cv)))
    mean_cv_dists = np.empty((216, len(partition[subset])))
    std_cv_dists     = np.empty((216, len(partition[subset])))
    mean_cv_absdists = np.empty((216, len(partition[subset])))
    std_cv_absdists  = np.empty((216, len(partition[subset])))
    bestacc     = 0
    std_bestacc = 0
    bestpatch   = 0

    for patch in patches:

        patch_name = patch
        print('At step ' + str(patch) + ' of ' + str(len(patches)))

        # Get map location
        i2 = int(patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6)
        i1 = int(np.floor((patch - np.floor(patch / 36) * 36) / 6))
        i0 = int(np.floor(patch / 36))

        try:
            patches.index(patch)
        except:
            missingPatches.append(patch)
            print('Missing patch ' + str(patch))
            map_Dist[i0, i1, i2, :] = 0
            map_stdDist[i0, i1, i2, :] = 0
            map_PredLabel[i0, i1, i2, :] = 0
            map_stdPredLabel[i0, i1, i2, :] = 0
            map_ACC[i0, i1, i2] = 0
            map_stdACC[i0, i1, i2] = 0
            continue

        counter = 0
        for c in range(0,len(cv)):
            dists    = np.empty((len(runs), len(partition[subset])))
            absdists = np.empty((len(runs), len(partition[subset])))
            accs     = []
            for r in range(0,len(runs)):

                if use3DorPI == '3D':

                    name_acc     = name + str(patch) + '_run' + str(runs[r]) + 'CV_' + str(cv[c]) +\
                                   '_tdadim[0'+tail_resAcc3D
                    path_acc     = os.path.join(output[0], str(patch), 'run_overviews', name_acc)
                    acc_df       = pd.read_csv(path_acc)

                    name_useDist = name + str(patch) + '_run' + str(runs[r]) + 'CV_' + str(cv[c]) +\
                                   '_tdadim[0'+tail_resdist3D
                    path_dist    = os.path.join(output[0], str(patch), 'run_distance', name_useDist)
                    dist_df      = pd.read_csv(path_dist)


                elif use3DorPI == 'PI':

                    name_acc     = prefixPI[4:]+ str(patch_name) + '_run' + str(runs[r]) + \
                                   'CV_' + str(cv[c])+suffix_PI[pi_dim]+str(pi_dim)+ tail_resAcc
                    name_useDist = prefixPI[4:]+ str(patch_name) + '_run' + str(runs[r]) + \
                                   'CV_' + str(cv[c])+suffix_PI[pi_dim]+str(pi_dim)+ tail_resdist

                    dist_df = pd.read_csv(os.path.join(output[0], str(patch),str(pi_dim), 'run_distance', name_useDist))
                    acc_df  = pd.read_csv(os.path.join(output[0], str(patch), str(pi_dim),'run_overviews', name_acc))
                else:
                    raise('Invalid value for use3DorPI!')

                accs.append(acc_df['acc'].values[0])
                counter = counter + 1

                for patID in range(0, len(partition[subset])):

                    if dist_df['thresh'].values[0] == 0:
                        absdists[r, patID] = 0
                        dists[r, patID] = 0
                    else:
                        absdists[r, patID] = dist_df.loc[patID, 'dist']

                        tmp    = (dist_df.loc[patID, 'y_pred'] - dist_df.loc[patID, 'thresh'])
                        thresh = dist_df.loc[patID, 'thresh']
                        if thresh == 0:
                            dist = 0
                        else:
                            if tmp < 0:
                                dist = tmp / thresh
                            else:
                                dist = tmp / (1 - thresh)
                        dists[r, patID] = dist


            # Average over runs
            mean_acc = np.nanmean(accs)
            std_acc  = np.nanstd(accs)
            for patID in range(0, len(partition[subset])):
                mean_dists[patch, patID, c] = np.nanmean(dists[:, patID])
                std_dists[patch, patID, c] = np.nanstd(dists[:, patID])
                mean_absdists[patch, patID, c] = np.nanmean(absdists[:, patID])
                std_absdists[patch, patID, c] = np.nanstd(absdists[:, patID])


        if counter == 0:
            missingPatches.append(patch)
            print('Missing patch ' + str(patch) + ' run/fold: '+str(runs[r])+str(cv[c]))
            map_Dist[i0, i1, i2, :] = 0
            map_stdDist[i0, i1, i2, :] = 0
            map_PredLabel[i0, i1, i2, :] = 0
            map_stdPredLabel[i0, i1, i2, :] = 0

        # Average over cv
        map_stdACC[i0, i1, i2] = std_acc
        map_ACC[i0, i1, i2]    = mean_acc
        if mean_acc >bestacc:
            bestacc = mean_acc
            std_bestacc = std_acc
            bestpatch = patch


        if mean_acc>bestacc:
            bestacc     = mean_acc
            std_bestacc = std_acc
            bestpatch   = patch

        for patID in range(0, len(partition[subset])):
            mean_cv_dists[patch, patID]    = np.mean(mean_dists[patch, patID, :])
            mean_cv_absdists[patch, patID] = np.mean(mean_absdists[patch, patID, :])

            if len(cv)>1:
                std_cv_dists[patch, patID]    = np.std(mean_dists[patch, patID, :])
                std_cv_absdists[patch, patID] = np.std(mean_absdists[patch, patID, :])
            else:
                std_cv_dists[patch, patID]    = std_dists[patch, patID, 0]
                std_cv_absdists[patch, patID] = std_absdists[patch, patID, 0]

            map_Dist[i0, i1, i2, patID] = mean_cv_absdists[patch, patID]
            map_stdDist[i0, i1, i2, patID] = std_cv_absdists[patch, patID]
            map_PredLabel[i0, i1, i2, patID] = mean_cv_dists[patch, patID]
            map_stdPredLabel[i0, i1, i2, patID] = std_cv_dists[patch, patID]

    print("Best acc: " + str(bestacc) + '+/-' + str(std_bestacc) + ' for patch ' + str(bestpatch))

    # Get missing patches:
    if len(missingPatches) > 0:
        print('Missing these patches: ', missingPatches)
    else:
        print('No missing patches')


    # Evaluate
    trueLabels = []
    for patID in range(0, len(partition[subset])):

        try:
            pat = partition[subset].values[patID,]
        except:
            pat = partition[subset][patID]
        trueLabel = labels[pat]
        trueLabels.append(trueLabel)

        # Get X:
        X[patID, :] = np.concatenate(
            (map_PredLabel[:, :, :, patID].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             map_stdPredLabel[:, :, :, patID].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             map_ACC[:, :, :].reshape((np.prod(map_PredLabel[:, :, :, patID].shape),)),
             ))

    return X, np.array(trueLabels)

def run_model(X_train, y_train, X_test, y_test, X_tst, y_tst,clf_choice, mydir, cv,r):

    # Three options of different models
    randomizer = False
    if clf_choice == 'LogisticRegression':
        clf = LogisticRegression(max_iter=10000, class_weight='balanced', solver='liblinear')
        param_grid = {'C': [0.001, 0.005, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'penalty': ['l1', 'l2']}
        randomizer = False
    else:
        print("Invalid Classifier!")

    ####Get classifier:
    print('Getting classifier')
    classifier = get_classifier(X_train, y_train, clf, param_grid, randomizer)

    #### Print coefficients (weights):
    if clf_choice == 'LogisticRegression':
        for coef, name in zip(classifier.best_estimator_.coef_.ravel(), X_train.columns):
            print(name, coef)

    print("Best parameters for " + clf_choice + ": ", classifier.best_params_)
    print("Best score for " + clf_choice + ": ", classifier.best_score_)

    ####
    # Evaluate training and validation performance
    plot_title = 'Evaluation of ' + clf_choice + ' model_CV'+str(cv)+'_run'+ str(r)
    y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn, thresh_trn =\
        evalBinaryClassifier_SimpleModel(classifier.best_estimator_, X_train, y_train, mydir,None,
                         plot_title=plot_title + '_train', doPlots=True)

    y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn,thresh_val  =\
        evalBinaryClassifier_SimpleModel(classifier.best_estimator_, X_test, y_test, mydir,thresh_trn,
                                          plot_title=plot_title + '_val', doPlots=True)

    y_tst, y_pred_class_tst, acc_tst, auc_tst, aps_tst, recall_tst,F1_tst,\
    precision_tst,tn_tst, tp_tst,fp_tst,fn_tst,thresh_tst  =\
        evalBinaryClassifier_SimpleModel(classifier.best_estimator_, X_tst, y_tst, mydir,thresh_trn,
                                          plot_title=plot_title + '_test', doPlots=True)


    return y_val, y_pred_class, acc, auc_val, aps, recall,F1,precision,tn, tp,fp,fn,\
           y_tst, y_pred_class_tst, acc_tst, auc_tst, aps_tst, recall_tst,F1_tst,precision_tst,\
           tn_tst, tp_tst,fp_tst,fn_tst

def evalBinaryClassifier_SimpleModel(model, x,y, mydir,thresh, labels=['Positives', 'Negatives'],
                         plot_title='Evaluation of model', doPlots=False, filename = ''):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the Recall-Precision curve, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    model : fitted scikit-learn model with predict_proba & predict methods
        and classes_ attribute. Typically LogisticRegression or
        LogisticRegressionCV

    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        in the data to be tested, and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target vector relative to x.

    labels: list, optional
        list of text labels for the two classes, with the positive label first

    Displays
    ----------
    4 Subplots

    Returns
    ----------
    F1: float
    '''


    # Generate predictions (probabilities -- the output of the last layer)
    y_pred = model.predict_proba(x)[:,1]


    # Optimize the threshold:
    if thresh is None:

        # Optimize the threshold:
        F1_opt = 0
        thresh_opt = 0
        threshs_test = list(np.arange(0, 1, 0.001)[500:])+list(np.arange(0, 1, 0.001)[:500])
        for t in threshs_test:
            y_pred_class = np.where(y_pred > t, 1, 0)
            cm = confusion_matrix(y, y_pred_class)
            tn, fp, fn, tp = [i for i in cm.ravel()]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * (precision * recall) / (precision + recall)
            if F1 > F1_opt:
                thresh_opt = t
                F1_opt = F1
    else:
        thresh_opt = thresh

    # Predict class
    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)

    # Get mean realtive distance to descision boundary
    dist = []
    for this_y in list(y_pred):
        if this_y-thresh_opt>0:
            distance = (this_y-thresh_opt)/(1-thresh_opt)
        else:
            distance = -(this_y - thresh_opt) / (thresh_opt)
        dist.append(distance)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_class)

    # Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y})

    # PRC:
    precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=1)
    aps = average_precision_score(y, y_pred)

    # ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y, y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # Plot
    if doPlots:
        fig = plt.figure(figsize=[20, 5])
        fig.suptitle(plot_title, fontsize=16)

        # 1 -- Confusion matrix
        plt.subplot(141)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                         annot_kws={"size": 14}, fmt='g')
        cmlabels = ['True Negatives', 'False Positives',
                    'False Negatives', 'True Positives']
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title('Confusion Matrix', size=15)
        plt.xlabel('Predicted Values', size=13)
        plt.ylabel('True Values', size=13)

        # 2 -- Distributions of Predicted Probabilities of both classes
        plt.subplot(142)
        plt.hist(df[df.target == 1].probPos, density=True, bins=25,
                 alpha=.5, color='green', label=labels[0])
        plt.hist(df[df.target == 0].probPos, density=True, bins=25,
                 alpha=.5, color='red', label=labels[1])
        plt.axvline(thresh_opt, color='blue', linestyle='--', label='Boundary')
        plt.xlim([0, 1])
        plt.title('Distributions of Predictions', size=15)
        plt.xlabel('Positive Probability (predicted)', size=13)
        plt.ylabel('Samples (normalized scale)', size=13)
        plt.legend(loc="upper right")

        # 3 -- PRC:
        plt.subplot(143)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.title('Recall-Precision Curve', size=15)
        plt.text(0.1, 0.3, f'AURPC = {round(aps, 2)}')
        plt.xlabel('Recall', size=13)
        plt.ylabel('Precision', size=13)
        plt.ylim([0.2, 1.05])
        plt.xlim([0.0, 1.0])

        # 4 -- ROC curve with annotated decision point
        plt.subplot(144)
        plt.plot(fp_rates, tp_rates, color='green',
                 lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

        # plot current decision point:
        plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', size=13)
        plt.ylabel('True Positive Rate', size=13)
        plt.title('ROC Curve', size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=.3)

        filenameuse = os.path.join(mydir, plot_title+filename + '.png')
        plt.savefig(filenameuse)
        plt.close('all')

    # F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return y, y_pred_class, acc, roc_auc, aps, recall,F1,precision,tn, tp,fp,fn,thresh_opt

def get_classifier(X_train, y_train, classifier, param_grid, randomizer=False):
    '''
    Create and fit a Logistic Regression model.
    Parameters:
    -----------
    X: data
    y: labels
    classifier: classifier (e.g. LogisticRegression())
    param_grid: Parameter Grid

    Returns:
    --------
    classifier
    X_test
    y_test
    '''
    if randomizer == False:
        classifier = GridSearchCV(classifier, param_grid, cv=5, verbose=1, scoring='roc_auc')  # score=...
    else:
        classifier = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=100, cv=3,
                                        verbose=1, random_state=42, n_jobs=-1,
                                        scoring='average_precision')  # 'roc_auc')

    classifier.fit(X_train, y_train)

    return classifier



if __name__ in "__main__":
    main()