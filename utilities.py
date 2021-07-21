###########################################################################################
# Imports
###########################################################################################

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score



###########################################################################################
# Functions
###########################################################################################

def evalBinaryClassifierCNN(model, x,y, mydir, labels=['Positives', 'Negatives'],
                         plot_title='Evaluation of model', doPlots=False, batchsize = 20,
                        filename = '', thresh_opt=None):
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

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x, y, batch_size=batchsize)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`

    y_pred = model.predict(x)[:,1]
    #y_pred_class = model.predict_classes(x)

    # Optimize the threshold:
    if thresh_opt is None:
        F1_opt = 0
        thresh_opt = 0
        for t in np.arange(0,1,0.01):
            y_pred_class  = np.where(y_pred > t, 1,0)
            cm = confusion_matrix(y[:, 1], y_pred_class)
            tn, fp, fn, tp = [i for i in cm.ravel()]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * (precision * recall) / (precision + recall)
            if F1>F1_opt:
                thresh_opt = t
                F1_opt = F1

    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)

    # get mean realtive distance to descision boundary
    dist = []
    for this_y in list(y_pred):
        if this_y-thresh_opt>0:
            distance = (this_y-thresh_opt)/(1-thresh_opt)
        else:
            distance = -(this_y - thresh_opt) / (thresh_opt)
        dist.append(distance)
    meanDist = np.mean(dist)
    stdDist = np.std(dist)



    # model predicts probabilities of positive class
    # p = model.predict_proba(x)
    # if len(model.classes_) != 2:
    #     raise ValueError('A binary class problem is required')
    # if model.classes_[1] == 1:
    #     pos_p = p[:, 1]
    #     pos_ind = 0
    # elif model.classes_[0] == 1:
    #     pos_p = p[:, 0]
    #     pos_ind = 0

    # 1 -- Confusion matrix
    cm = confusion_matrix(y[:,1], y_pred_class)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y[:,1]})

    # 3 -- PRC:
    #y_score = model.predict_proba(x)[:, pos_ind]
    precision, recall, _ = precision_recall_curve(y[:,1], y_pred, pos_label=1)
    aps = average_precision_score(y[:,1], y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y[:,1], y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # FIGURE
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
        # #plt.show(block=False)

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
        #plt.show(block=False)

        filenameuse = os.path.join(mydir, plot_title+filename + '.png')
        plt.savefig(filenameuse)
        plt.close('all')

    # 5 -- F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)



    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return tn, fp, fn, tp,acc, precision,recall,roc_auc,aps, dist,meanDist,stdDist, thresh_opt, y_pred

def evalBinaryClassifierCNN_datagen(model, val_generator, mydir, labels=['Positives', 'Negatives'],
                         plot_title='Evaluation of model', doPlots=False, batchsize = 20,
                        filename = '', thresh_opt=None):
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

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    # results = model.evaluate(x, y, batch_size=batchsize)
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #y = val_generator.__getall_Y__()

    X,y = val_generator.__getall__()
    print('Y: '+str(y.shape))

    # if bs>100:
    #     bs = batchsize
    #print('Batch size:'+str(bs))
    y_pred = model.predict(X, batch_size=batchsize)[:,1]
    print('Y_pred: '+str(y_pred.shape))
    #y_pred_class = model.predict_classes(x)

    # Optimize the threshold:
    if thresh_opt is None:
        F1_opt = 0
        thresh_opt = 0
        for t in np.arange(0,1,0.01):
            y_pred_class  = np.where(y_pred > t, 1,0)
            cm = confusion_matrix(y[:, 1], y_pred_class)
            tn, fp, fn, tp = [i for i in cm.ravel()]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * (precision * recall) / (precision + recall)
            if F1>F1_opt:
                thresh_opt = t
                F1_opt = F1

    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)

    # get mean realtive distance to descision boundary
    dist = []
    for this_y in list(y_pred):
        if this_y-thresh_opt>0:
            distance = (this_y-thresh_opt)/(1-thresh_opt)
        else:
            distance = -(this_y - thresh_opt) / (thresh_opt)
        dist.append(distance)
    meanDist = np.mean(dist)
    stdDist = np.std(dist)



    # model predicts probabilities of positive class
    # p = model.predict_proba(x)
    # if len(model.classes_) != 2:
    #     raise ValueError('A binary class problem is required')
    # if model.classes_[1] == 1:
    #     pos_p = p[:, 1]
    #     pos_ind = 0
    # elif model.classes_[0] == 1:
    #     pos_p = p[:, 0]
    #     pos_ind = 0

    # 1 -- Confusion matrix
    cm = confusion_matrix(y[:,1], y_pred_class)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y[:,1]})

    # 3 -- PRC:
    #y_score = model.predict_proba(x)[:, pos_ind]
    precision, recall, _ = precision_recall_curve(y[:,1], y_pred, pos_label=1)
    aps = average_precision_score(y[:,1], y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y[:,1], y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # FIGURE
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
        # #plt.show(block=False)

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
        #plt.show(block=False)

        filenameuse = os.path.join(mydir, plot_title+filename + '.png')
        plt.savefig(filenameuse)
        plt.close('all')

    # 5 -- F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)



    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return tn, fp, fn, tp,acc, precision,recall,roc_auc,aps, dist,meanDist,stdDist, thresh_opt, y_pred

def getArrayWithTest(data_folder_all, patch, partition, labels, dim, n_channels, usescale01 = False):

    if patch<217:
        folder = os.path.join(data_folder_all,str(patch))
    else:
        folder = data_folder_all

    X_trn = np.empty((len(partition['train']), *dim, n_channels))
    y_trn = np.empty((len(partition['train'])), dtype=int)
    X_val = np.empty((len(partition['validation']), *dim, n_channels))
    y_val = np.empty((len(partition['validation'])), dtype=int)
    X_tst = np.empty((len(partition['test']), *dim, n_channels))
    y_tst = np.empty((len(partition['test'])), dtype=int)

    # Generate data
    for i, ID in enumerate(partition['train']):
        if i ==0:
            print('at step %d of %d' % (i, len(partition['train'])))

        # load
        sc_X = np.load(os.path.join(folder, ID + '.npy'))

        # Store sample
        if len(sc_X.shape)<4:
            X_trn[i,] = np.expand_dims(sc_X, axis=3)
        else:
            X_trn[i,] = sc_X

        # Store class
        y_trn[i]  = labels[ID]

    for i, ID in enumerate(partition['validation']):
        if i == 0:
            print('at step %d of %d' % (i, len(partition['validation'])))

        # load
        sc_X = np.load(os.path.join(folder, ID + '.npy'))

        # Store sample
        if len(sc_X.shape)<4:
            X_val[i,] = np.expand_dims(sc_X, axis=3)
        else:
            X_val[i,] = sc_X

        # Store class
        y_val[i]  = labels[ID]

    for i, ID in enumerate(partition['test']):
        if i == 0:
            print('at step %d of %d' % (i, len(partition['test'])))

        # load
        sc_X = np.load(os.path.join(folder, ID + '.npy'))

        # Store sample
        if len(sc_X.shape)<4:
            X_tst[i,] = np.expand_dims(sc_X, axis=3)
        else:
            X_tst[i,] = sc_X

        # Store class
        y_tst[i]  = labels[ID]


    return X_trn, y_trn, X_val, y_val, X_tst, y_tst


def create_split(
        data_dir='PIdir',
        dimensions=[0, 1, 2],
        partition_file='partitionFile',
        labels_file='yourLabels',
        reshape=False,
        resolution=[50, 50],
        use_T1w=False,
        normalize=False
):
    part = np.load(partition_file, allow_pickle=True).item()
    labels = np.load(labels_file, allow_pickle=True).item()
    X_dict = {}
    y_dict = {}

    for key in part.keys():
        X_dict[key] = []
        y_dict[key] = []

        for composite in part[key]:
            comp_split = composite.split('-')
            subject = comp_split[0] + '-' + comp_split[1]
            session = comp_split[2]
            mode = comp_split[3]
            if use_T1w:
                mode = 'T1w'

            sess_dir = os.path.join(data_dir, subject, session)
            pi_list = [x for x in os.listdir(sess_dir) if (mode in x)]

            for d in [0, 1, 2]:  # exclude dimensions if desired:
                if d not in dimensions:
                    pi_list = [x for x in pi_list if f'dim{d}' not in x]

            next_pi = []
            for pi_im in pi_list:

                # get the persistence image of the subject & session:
                with open(os.path.join(sess_dir, pi_im), 'r') as pimg_data:
                    pi = json.load(pimg_data)
                if reshape:
                    try:
                        next_pi.append(np.array(pi['0'][0]).reshape(resolution[0], resolution[1]))
                    except:
                        print('No PI!')
                        empty_PI = np.zeros((resolution[0], resolution[1]))
                        next_pi.append(empty_PI)
                else:
                    next_pi.extend(pi['0'][0])
            if reshape:
                try:
                    next_pi = np.transpose(np.array(next_pi), (1, 2, 0))
                except:
                    print('ERROR')

            if normalize:
                meanPI = np.mean(next_pi.flatten())
                stdPI = np.std(next_pi.flatten())
                next_pi = (next_pi - meanPI) / stdPI
                print('Normalized PI!')

            X_dict[key].append(next_pi)
            y_dict[key].append(labels[composite])

    return X_dict, y_dict


# For GNN
def evaluateFinal_Notlin(predictions_in, labels, thresh, mydir, name, useBCE):
    #model.eval()

    # Convert batches:
    predictions = predictions_in[0]
    for i in range(1, len(predictions_in)):
        predictions = np.concatenate((predictions, predictions_in[i]))
    predictions = predictions[:,1]

    if useBCE:
        predictions = np.exp(predictions)
        print('Prediction: ')
        print(predictions)

    if thresh is None:
        thresh = optThresh(labels, predictions)
    print(thresh)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pred_classes = []


    for i in range(0,len(predictions)):

        pred_class = np.where(predictions[i] > thresh, 1, 0)
        pred_classes.append(pred_class)
        try:
            label = labels[i]
        except:
            print(i)
            label = labels[i]
        if np.logical_and(label == 0, pred_class  == 0):
            tn += 1
        elif np.logical_and(label == 0, pred_class  == 1):
            fp += 1
        elif np.logical_and(label == 1, pred_class  == 0):
            fn += 1
        elif np.logical_and(label == 1, pred_class  == 1):
            tp += 1

    y_pred = predictions
    y_pred_class = np.array(pred_classes)
    y = np.array(labels)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y})

    # 3 -- PRC:
    #y_score = model.predict_proba(x)[:, pos_ind]
    precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=1)
    aps = average_precision_score(y, y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y, y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)

    cm = np.array([[tn, fp],[fn,tp]])

    # FIGURE
    fig = plt.figure(figsize=[20, 5])
    fig.suptitle(name, fontsize=16)

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
             alpha=.5, color='green', label='1')
    plt.hist(df[df.target == 0].probPos, density=True, bins=25,
             alpha=.5, color='red', label='0')
    plt.axvline(thresh, color='blue', linestyle='--', label='Boundary')
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
    # #plt.show(block=False)

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
    # plt.show(block=False)

    filenameuse = os.path.join(mydir,'run_eval', name +'.png')
    plt.savefig(filenameuse)
    plt.close('all')
    print(filenameuse)

    # 5 -- F1 score
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except:
        recall = 0

    try:
        F1 = 2 * (precision * recall) / (precision + recall)
    except:
        F1 = 0

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return y, y_pred_class, acc, roc_auc, aps, recall,F1,precision,tn, tp,fp,fn,thresh
def get_patchgraph(patches):
    sources = []
    targets = []
    for p in patches:

        # Get neighbours
        neighbours = getpatchneighbours(p, patches)

        # Convert to nodes
        neighbour_nodes = [patches.index(n) for n in neighbours]

        # Graph
        sources = sources + list(patches.index(p) * np.ones((len(neighbours),)))
        targets = targets + neighbour_nodes

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return edge_index
def getpatchneighbours(patch, patches):
    # Get map location
    i0, i1, i2 = patch2ind(patch)

    neighbours = []
    for j2 in [-1, 0, 1]:
        for j1 in [-1, 0, 1]:
            for j0 in [-1, 0, 1]:

                # each patch has up to 26 neighbours
                if not j0 == j1 == j2 == 0:
                    this_patch = ind2patch(j0 + i0, j1 + i1, j2 + i2)

                    if this_patch in patches:
                        neighbours.append(this_patch)

    return neighbours
def ind2patch(j0, j1, j2):
    patch = j0 * 36 + j1 * 6 + j2
    return patch
def patch2ind(patch):
    i2 = int(patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6)
    i1 = int(np.floor((patch - np.floor(patch / 36) * 36) / 6))
    i0 = int(np.floor(patch / 36))
    return i0, i1, i2
def getInnerPatches():
    ids = np.zeros((6, 6, 6))
    innerPatch = []
    for patch in range(0, 216):
        patchid2 = patch - np.floor(patch / 36) * 36 - np.floor((patch - np.floor(patch / 36) * 36) / 6) * 6
        patchid1 = np.floor((patch - np.floor(patch / 36) * 36) / 6)
        patchid0 = np.floor(patch / 36)
        ids[int(patchid0), int(patchid1), int(patchid2)] = patch
        #print('Patch ' + str(patch) + ': ' + str(patchid0) + ' ' + str(patchid1) + ' ' + str(patchid2))

        if patchid0 > 0 and patchid0 < 5 and patchid1 > 0 and patchid1 < 5 and patchid2 > 0 and patchid2 < 5:
            innerPatch.append(patch)

    return innerPatch