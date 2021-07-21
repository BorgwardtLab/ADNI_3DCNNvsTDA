###########################################################################################
# Libraries
###########################################################################################
import os
import numpy as np
from pathlib import Path
import pandas as pd
import argparse


###########################################################################################
# Helper Functions - Please edit with relevant paths and folder names
###########################################################################################

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

def main_encNew(c,path):

    folder               = path + 'output_TDA/NewPatches'
    suffix               = 'yourSuffix'
    model_name           = 'PI_CNN_model_None_dense_patch'
    featureFolderAllDims = 'yourfolder'
    useWeighting         = True
    runs                 = [0,1,2]
    cvs                  = [c]
    n_enc                = 32
    dims                 = [0,1,2]
    patches              = getInnerPatches()

    # Loop over TDA dimensions
    for dim in dims:

        featureFolder = featureFolderAllDims+str(dim)

        for c in cvs:

            print("At cv fold "+str(c))

            # Get partition
            partition_suffix = '0CN_1AD_pat_1GO23_ML4HCsplitWithTest_'+ str(c)
            partition_file   = path + 'partitions/partition_' + partition_suffix + '.npy'
            partition        = np.load(partition_file, allow_pickle='TRUE').item()
            label_file       = path + 'partitions/labels_' + partition_suffix + '.npy'
            labels           = np.load(label_file, allow_pickle='TRUE').item()
            labels_trn       = [labels[id] for id in partition['train']]


            # Prapare folder
            saveToFolder = os.path.join(folder, featureFolder, str(c))
            Path(saveToFolder).mkdir(parents=True, exist_ok=True)

            for r in runs:
                print("At run " + str(r))

                for j,patch in enumerate(patches):
                    print("At patch " + str(j)+' of '+str(len(patches)))
                    patch_Name = patch

                    # Encodings
                    filepath_enc_trn = os.path.join(os.path.join(folder, str(patch),str(dim),'run_hdf5'),
                                                    'encoding_trn_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str( c) +
                                                    '_tdadim['+str(dim)+']' + suffix+'.npy')
                    with open(filepath_enc_trn, 'rb') as f:
                        enc_trn = np.load(f, allow_pickle=True)

                    filepath_enc_val = os.path.join(os.path.join(folder,str(patch),str(dim), 'run_hdf5'),
                                                    'encoding_val_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str(
                                                        c) + '_tdadim['+str(dim)+']' + suffix+'.npy')
                    with open(filepath_enc_val, 'rb') as f_val:
                        enc_val = np.load(f_val, allow_pickle=True)

                    filepath_enc_tst = os.path.join(os.path.join(folder, str(patch),str(dim), 'run_hdf5'),
                                                    'encoding_tst_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str(
                                                        c) + '_tdadim[' + str(dim) + ']' + suffix+'.npy')
                    with open(filepath_enc_tst, 'rb') as f_tst:
                        enc_tst = np.load(f_tst, allow_pickle=True)


                    # weights
                    filepath_weights = os.path.join(os.path.join(folder, str(patch),str(dim), 'run_hdf5'),
                                                    'encoding_weights_preClassLayer'+model_name
                                                    + str(patch_Name) +'_run' + str(r) + 'CV_' + str(c) +
                                                    '_tdadim['+str(dim)+']l1_tail0.001_do0.5_ksize4_nlayers4_lr0.0001_bs50.npy')
                    with open(filepath_weights, 'rb') as f:
                        weights = np.load(f, allow_pickle=True)
                        trainWeights = weights[0][:,1]
                        bias         = weights[1][1]

                    # Descision boundary
                    filepath_thresh_trn = os.path.join(os.path.join(folder, str(patch),str(dim), 'run_distance'),
                                                    model_name + str(patch_Name) +'_run' + str(r) + 'CV_' + str(c) +
                                                    '_tdadim['+str(dim)+']'+ suffix+'.csv')
                    thresh = pd.read_csv(filepath_thresh_trn)['thresh'][0]


                    # Restructure based on highest to lowest variance
                    inds0 = np.where(np.asarray(labels_trn) == 0)
                    inds1 = np.where(np.asarray(labels_trn) == 1)
                    var0  = np.var(enc_trn[inds0,:].squeeze(),axis = 0)
                    var1  = np.var(enc_trn[inds1,:].squeeze(),axis = 0)
                    var01 = np.var(enc_trn[:,:],axis = 0)
                    score = (var0+var1)/var01
                    ind   = np.argsort(score)
                    enc_trnsort = enc_trn[:, ind]
                    enc_valsort = enc_val[:, ind]
                    enc_tstsort = enc_tst[:, ind]


                    # Loop over all training patients
                    for i,p in enumerate(partition['train']):
                        name = p+'_cv'+ str(c)+'_run'+str(r)+'_patch'+str(patch)+ '.npy'

                        if useWeighting:
                            data = enc_trnsort[i,:]*trainWeights[ind] + bias/n_enc - np.log(thresh/(1-thresh))
                        else:
                            data = enc_trnsort[i, :]
                        np.save(os.path.join(saveToFolder,name), data)

                    # Loop over all validation patients
                    for i,p in enumerate(partition['validation']):
                        name = p + '_cv' + str(c) + '_run' + str(r)+'_patch'+str(patch)+ '.npy'
                        if useWeighting:
                            data = enc_valsort[i, :] * trainWeights[ind] + bias / n_enc - np.log(thresh / (1 - thresh))
                        else:
                            data = enc_valsort[i, :]
                        np.save(os.path.join(saveToFolder, name), data)

                    # Loop over all validation patients
                    for i, p in enumerate(partition['test']):
                        name = p + '_cv' + str(c) + '_run' + str(r) + '_patch' + str(patch) + '.npy'
                        if useWeighting:
                            data = enc_tstsort[i, :] * trainWeights[ind] + bias / n_enc - np.log(
                                thresh / (1 - thresh))
                        else:
                            data = enc_tstsort[i, :]
                        np.save(os.path.join(saveToFolder, name), data)

def main_encNew3D(c,path):

    folder        = path + 'output_3D/PatchesNEWENC'
    featureFolderAllDims = 'yourfolder'
    model_name     = 'None_CNNflexi_dense_patch'
    partition_name = '0CN_1AD_pat_1GO23_ML4HC_'
    useWeighting   = True
    runs           = [0,1,2]
    cvs            = [c]
    patches        = getInnerPatches()
    n_enc          = 32
    dims           = [0] # Not multiple dims for 3D models
    for dim in dims:
        featureFolder = featureFolderAllDims+str(dim)

        for c in cvs:

            print("At cv fold "+str(c))

            # Get partition
            partition_suffix = partition_name+ str(c)
            partition_file   = path + 'partitions/partition_' + partition_suffix + '.npy'
            partition        = np.load(partition_file, allow_pickle='TRUE').item()
            label_file       = path + 'partitions/labels_' + partition_suffix + '.npy'
            labels           = np.load(label_file, allow_pickle='TRUE').item()
            labels_trn       = [labels[id] for id in partition['train']]

            # Prapare folder
            saveToFolder = os.path.join(folder, featureFolder, str(c))
            Path(saveToFolder).mkdir(parents=True, exist_ok=True)

            # Loop over all runs
            for r in runs:
                print("At run " + str(r))

                # Loop over all patches
                for j,patch in enumerate(patches):
                    print("At patch " + str(j)+' of '+str(len(patches)))
                    patch_Name = patch


                    # Encodings
                    filepath_enc_trn = os.path.join(os.path.join(folder, str(patch),'run_hdf5'),
                                                    'encoding_trn_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str( c) +
                                                    '_tdadim['+str(dim)+'].npy')
                    with open(filepath_enc_trn, 'rb') as f:
                        enc_trn = np.load(f, allow_pickle=True)

                    filepath_enc_val = os.path.join(os.path.join(folder,str(patch), 'run_hdf5'),
                                                    'encoding_val_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str(
                                                        c) + '_tdadim['+str(dim)+'].npy')
                    with open(filepath_enc_val, 'rb') as f_val:
                        enc_val = np.load(f_val, allow_pickle=True)

                    filepath_enc_tst = os.path.join(os.path.join(folder, str(patch), 'run_hdf5'),
                                                    'encoding_tst_'+model_name + str(patch_Name) +
                                                    '_run' + str(r) + 'CV_' + str(
                                                        c) + '_tdadim[' + str(dim) + '].npy')
                    with open(filepath_enc_tst, 'rb') as f_tst:
                        enc_tst = np.load(f_tst, allow_pickle=True)


                    # weights
                    filepath_weights = os.path.join(os.path.join(folder, str(patch), 'run_hdf5'),
                                                    'encoding_weights_preClassLayer'+model_name
                                                    + str(patch_Name) +'_run' + str(r) + 'CV_' + str(c) +
                                                    '_tdadim['+str(dim)+'].npy')
                    with open(filepath_weights, 'rb') as f:
                        weights = np.load(f, allow_pickle=True)
                        trainWeights = weights[0][:,1]
                        bias         = weights[1][1]

                    # Descision boundary
                    filepath_thresh_trn = os.path.join(os.path.join(folder, str(patch), 'run_distance'),
                                                    model_name + str(patch_Name) +'_run' + str(r) + 'CV_' + str(c) +
                                                    '_tdadim['+str(dim)+']_results_dist_trn.csv')
                    thresh = pd.read_csv(filepath_thresh_trn)['thresh'][0]



                    # Restructure based on highest to lowest variance
                    inds0 = np.where(np.asarray(labels_trn) == 0)
                    inds1 = np.where(np.asarray(labels_trn) == 1)
                    var0  = np.var(enc_trn[inds0,:].squeeze(),axis = 0)
                    var1  = np.var(enc_trn[inds1,:].squeeze(),axis = 0)
                    var01 = np.var(enc_trn[:,:],axis = 0)
                    score = (var0+var1)/var01
                    ind   = np.argsort(score)
                    enc_trnsort = enc_trn[:, ind]
                    enc_valsort = enc_val[:, ind]
                    enc_tstsort = enc_tst[:, ind]


                    # Loop over all training patients
                    for i,p in enumerate(partition['train']):
                        name = p+'_cv'+ str(c)+'_run'+str(r)+'_patch'+str(patch)+ '.npy'

                        if useWeighting:
                            data = enc_trnsort[i,:]*trainWeights[ind] + bias/n_enc - np.log(thresh/(1-thresh))
                        else:
                            data = enc_trnsort[i, :]
                        np.save(os.path.join(saveToFolder,name), data)

                    # Loop over all validation patients
                    for i,p in enumerate(partition['validation']):
                        name = p + '_cv' + str(c) + '_run' + str(r)+'_patch'+str(patch)+ '.npy'
                        if useWeighting:
                            data = enc_valsort[i, :] * trainWeights[ind] + bias / n_enc - np.log(thresh / (1 - thresh))
                        else:
                            data = enc_valsort[i, :]
                        np.save(os.path.join(saveToFolder, name), data)

                    # Loop over all validation patients
                    for i, p in enumerate(partition['test']):
                        name = p + '_cv' + str(c) + '_run' + str(r) + '_patch' + str(patch) + '.npy'
                        if useWeighting:
                            data = enc_tstsort[i, :] * trainWeights[ind] + bias / n_enc - np.log(
                                thresh / (1 - thresh))
                        else:
                            data = enc_tstsort[i, :]
                        np.save(os.path.join(saveToFolder, name), data)










                    #######################

if __name__ in "__main__":

    id   = 1 # 0: TDA based input 1: 3D CNN based input
    path = 'yourPath'

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', nargs='+', type=int, default=91)
    args = parser.parse_args()
    cv  = args.cv[0]


    if id == 0:
        main_encNew3D(cv, path)
    elif id == 1:
        main_encNew(cv)