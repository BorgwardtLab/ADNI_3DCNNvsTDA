import argparse
from pathlib import Path
import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger, TestTubeLogger

from utilities import get_patchgraph, getInnerPatches, evaluateFinal_Notlin

# Check whether we have a GPU device
GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0


class GNNLayer(nn.Module):
    """Simple GNN layer with configurable details."""
    def __init__(
        self,
        in_features,
        out_features,
        p_dropout=0.0,
        batch_norm=False,
        activation=nn.ReLU(),
        residual=False,
        layer_type='GCN',
    ):
        super().__init__()


        # Layers
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = nn.Identity()
        self.activation = activation
        self.dropout    = nn.Dropout(p_dropout)
        self.residual   = residual
        self.layer_type = layer_type
        self.layer      = GCNConv(in_features, out_features, add_self_loops=False)


    def forward(self, x, edge_index):
        h = self.layer(x, edge_index)
        h = self.batch_norm(h)
        h = self.activation(h)

        if self.residual:
            h = h + x

        return self.dropout(h)

class GNNModel(pl.LightningModule):
    """Simple GNN model with a certain number of layers."""

    def __init__(
        self,
        hidden_dim,
        depth,
        n_node_features,
        weights,
        n_classes=2,
        lr = 0.0001,
        p_dropout=0.25,
        patience=100,
        dimensions=[0, 1, 2],
        batch_norm=True,
        layer_type='GCN'
    ):
        super().__init__()
        self.save_hyperparameters()
        self.weights    = weights
        self.n_classes  = n_classes
        self.patience   = patience
        self.lr         = lr
        self.dimensions = dimensions
        self.num_dims   = len(dimensions)
        self.batch_norm = batch_norm
        self.layer_type = layer_type
        self.embedding  = torch.nn.Linear(self.num_dims * n_node_features, hidden_dim)

        self.layers = nn.ModuleList([
            GNNLayer(
                hidden_dim,
                hidden_dim,
                p_dropout=p_dropout,
                batch_norm=self.batch_norm,
                layer_type=self.layer_type,
            ) for _ in range(depth)
        ])

        self.pooling_fn = global_mean_pool

        # Classification layer.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_classes),
            torch.nn.Softmax(dim=1)
        )

        # optimization
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=5, factor=0.5)

        # Monitor and evaluate
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def configure_optimizers(self):

        scheduler = {
                     'scheduler': self.lr_scheduler,
                     'monitor': 'val_loss',
                     'interval': 'epoch',
                     'frequency': 1
        }
        return [self.optimizer], [scheduler]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, edge_index=edge_index, data=data)

        x = self.pooling_fn(x, data.batch)
        x = self.classifier(x)

        return x

    def training_step(self, batch):
        y     = batch.y.long()
        y_hat = self(batch)
        loss  = self.loss(y_hat.float().squeeze(), y)
        y_hat = torch.argmax(y_hat, dim=1)

        self.train_accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)


        return loss

    def validation_step(self, batch):
        y     = batch.y.long()
        y_hat = self(batch)
        loss  = self.loss(y_hat.float().squeeze(), y)
        y_hat = torch.argmax(y_hat, dim=1)

        self.val_accuracy(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, on_epoch=True, on_step=True, prog_bar=True)
        return {
            'predictions': y_hat.detach().cpu(),
            'labels': y.detach().cpu()
        }

    def test_step(self, batch):
        y     = batch.y.long()
        y_hat = self(batch)
        loss  = self.loss(y_hat.float().squeeze(), y)
        y_hat = torch.argmax(y_hat, dim=1)

        self.test_accuracy(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser],
            add_help=False
        )

        parser.add_argument('--depth', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--p_dropout', type=float, default=0)

        return parser

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, partition, labels, patches, edge_index, folder, featureFolder, c, r, n_features, dimensions=[0, 1, 2], n_workers=32):
        super().__init__()
        self.batch_size = batch_size
        self.partition = partition
        self.patches = patches
        self.edge_index = edge_index
        self.folder = folder
        self.featureFolder = featureFolder
        self.n_features = n_features
        self.c = c
        self.r = r
        self.labels = labels
        self.labels_trn = [labels[id] for id in partition['train']]
        self.labels_val = [labels[id] for id in partition['validation']]
        self.labels_tst = [labels[id] for id in partition['test']]
        self.len_val = len(self.labels_val)
        self.dimensions = dimensions
        self.n_workers = n_workers

    def setup(self, stage=None):

        # Loop over all training patients
        data_trn = []
        for i, p in enumerate(self.partition['train']):
            len_dims = len(self.dimensions)
            x = np.zeros((len(self.patches), len_dims * self.n_features))
            y = self.labels[p]

            for i, patch in enumerate(self.patches):
                name = p + '_cv' + str(self.c) + '_run' + str(self.r) + '_patch' + str(patch) + '.npy'
                x_alldims = np.array([])

                for dim in self.dimensions:                    
                    filename = os.path.join(self.folder, self.featureFolder+str(dim), str(self.c), name)
                    thisx = np.load(filename)
                    x_alldims = np.concatenate((x_alldims, thisx))

                if not np.isnan(x_alldims).any() and not np.isinf(x_alldims).any():
                    x[i, :] = x_alldims
                else:
                    continue
                
            data = Data(x=torch.Tensor(x), edge_index=self.edge_index, y=torch.from_numpy(np.asarray(y)))
            data_trn.append(data)
                
        # Loop over all validation patients
        data_val = []
        for i, p in enumerate(self.partition['validation'][:]):
            len_dims = len(self.dimensions)
            x = np.zeros((len(self.patches), len_dims * self.n_features))
            y = self.labels[p]

            for i, patch in enumerate(self.patches):
                name = p + '_cv' + str(self.c) + '_run' + str(self.r) + '_patch' + str(patch) + '.npy'
                x_alldims = np.array([])

                for dim in self.dimensions:
                    filename = os.path.join(self.folder, self.featureFolder+str(dim), str(self.c), name)
                    thisx = np.load(filename)
                    x_alldims = np.concatenate((x_alldims, thisx))

                if not np.isnan(x_alldims).any() and not np.isinf(x_alldims).any():
                    x[i, :] = x_alldims
                else:
                    continue

            data = Data(x=torch.Tensor(x), edge_index=self.edge_index, y=torch.from_numpy(np.asarray(y)))
            data_val.append(data)

        # Loop over all test patients
        data_tst = []
        for i, p in enumerate(self.partition['test'][:]):
            len_dims = len(self.dimensions)
            x = np.zeros((len(self.patches), len_dims * self.n_features))
            y = self.labels[p]

            for i, patch in enumerate(self.patches):
                name = p + '_cv' + str(self.c) + '_run' + str(self.r) + '_patch' + str(patch) + '.npy'
                x_alldims = np.array([])

                for dim in self.dimensions:
                    filename = os.path.join(self.folder, self.featureFolder + str(dim), str(self.c), name)
                    thisx = np.load(filename)
                    x_alldims = np.concatenate((x_alldims, thisx))

                if not np.isnan(x_alldims).any() and not np.isinf(x_alldims).any():
                    x[i, :] = x_alldims
                else:
                    continue

            data = Data(x=torch.Tensor(x), edge_index=self.edge_index, y=torch.from_numpy(np.asarray(y)))
            data_tst.append(data)

        self.data_trn = data_trn
        self.data_val = data_val
        self.data_tst = data_tst

    def train_dataloader(self):
        loader_trn = DataLoader(
            self.data_trn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader_trn

    def train_dataloader_eval(self):
        loader_trn = DataLoader(
            self.data_trn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader_trn

    def val_dataloader(self):
        loader_val = DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader_val

    def test_dataloader(self):
        return DataLoader(
            self.data_tst,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=False,
            pin_memory=True
        )


if __name__ == '__main__':

    # Model inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--max_epochs', type=int, default=1500)
    parser.add_argument('--runs', nargs='+', type=int, default=[0])
    parser.add_argument('--cv', nargs='+', type=int, default=[1])
    parser     = GNNModel.add_model_specific_args(parser)
    args       = parser.parse_args()
    bs         = args.batch_size
    num_epochs = args.max_epochs
    r          = args.runs[0]
    c          = args.cv[0]
    patience   = 100
    batch_norm = True
    n_features = 32
    n_classes  = 2
    dimensions = [0, 1, 2]

    # Paths and folders
    path             = 'yourPath'
    folder           = path + 'output_TDA/Patches/'
    featureFolder    = os.path.join(folder, 'nodeFeaturesWeightesNEWCV5_dim')
    layer_type       = 'GCN'
    the_time         = str(time.time())
    output           = path + f'ML4HCruns/output_TDA/Patches_splitswithTest/GNN_encodings/{layer_type}'#f'ML4HCruns/output_TDA/Patches/GNN_encodings/{layer_type}'
    model_name       = f'{layer_type}_test_cv'+str(c) + '_run'+str(r) + f'_bs{bs}_do{args.p_dropout}_depth{args.depth}_lr{args.lr}'
    tt_logger_name   = f'lightning_TT_logs/{model_name}'+the_time
    tb_logger_name   = f'lightning_TB_logs/{model_name}'+the_time
    tt_plot_save_dir = os.path.join(output, tt_logger_name)
    logger_name      = 'TT' # Options: 'TT' (test tube) or 'TB' (tensorboard)
    partition_name   = '0CN_1AD_pat_1GO23_ML4HC_'
    

    # Create folders if needed
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_distance')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_eval')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_hdf5')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_losses')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_overviews')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'run_parameters')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output, 'figures')).mkdir(parents=True, exist_ok=True)

    # Get partition
    partition_suffix = partition_name + str(c)
    partition_file   = path + 'partitions/partition_' + partition_suffix + '.npy'
    partition        = np.load(partition_file, allow_pickle='TRUE').item()
    label_file       = path + 'partitions/labels_' + partition_suffix + '.npy'
    labels           = np.load(label_file, allow_pickle='TRUE').item()
    labels_trn       = [labels[id] for id in partition['train']]
    labels_val       = [labels[id] for id in partition['validation']]

    # Graph
    patches    = getInnerPatches()
    edge_index = get_patchgraph(patches)
    device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    weights = torch.Tensor([1 - np.sum(labels_trn) / len(labels_trn), np.sum(labels_trn) / len(labels_trn)]).to(device)
    
    # Callbacks
    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=100
    )
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='max',
        verbose=True
    )
    tb_logger = TensorBoardLogger(
        save_dir=output,
        name=tb_logger_name,
    )
    
    tt_logger = TestTubeLogger(
        save_dir=output,
        name=tt_logger_name,
        )
    logger = tt_logger
    if logger_name == 'TB':
        logger = tb_logger

    # Prepare training
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb],
        logger=logger,
        default_root_dir=os.path.join(output,'run_hdf5'),
    )

    # Get data
    data = DataModule(
        batch_size=bs,
        partition=partition,
        labels=labels,
        patches=patches,
        edge_index=edge_index,
        folder=folder,
        featureFolder=featureFolder,
        c=c,
        r=r,
        n_features=n_features,
        dimensions=dimensions,
        n_workers=3,
    )
 

    # The model
    model = GNNModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        weights=weights,
        n_node_features=data.n_features,
        n_classes=n_classes,
        lr=args.lr,
        p_dropout=args.p_dropout,
        dimensions=dimensions,
        batch_norm=batch_norm,
        layer_type=layer_type,
    )

    # Train
    trainer.fit(model, datamodule=data)

    # Restore the best model and use it for validation / testing.
    checkpoint_path = checkpoint_cb.best_model_path
    trainer_best = pl.Trainer(logger=False)
    model = GNNModel.load_from_checkpoint(checkpoint_path)

    # Predict all sets
    predict_trn = trainer.predict(dataloaders=data.train_dataloader_eval())
    predict_val = trainer.predict(dataloaders=data.val_dataloader())
    predict_tst = trainer.predict(dataloaders=data.test_dataloader())

    # Save predictions
    y_trn, y_pred_class_trn, acc_trn, roc_auc_trn, aps_trn, recall_trn, F1_trn, \
    precision_trn, tn_trn, tp_trn, fp_trn, fn_trn, thresh_opt = \
        evaluateFinal_Notlin(predict_trn, data.labels_trn[:len(predict_trn) * bs], None, output, model_name, False)
    y_val, y_pred_class_val, acc, roc_auc, aps, recall, F1, \
    precision, tn, tp, fp, fn, thresh_opt = \
        evaluateFinal_Notlin(predict_val, data.labels_val, thresh_opt, output, 'val_' + model_name, False)
    y_tst, y_pred_class_tst, acc_tst, roc_auc_tst, aps_tst, recall_tst, F1_tst, \
    precision_tst, tn_tst, tp_tst, fp_tst, fn_tst, thresh_opt = \
        evaluateFinal_Notlin(predict_tst, data.labels_tst[:len(predict_tst) * bs], thresh_opt, output, 'tst_' + model_name, False)


    # Create data frame
    data_df = {'runID': str(r),
               'n_epochs': 0,
               'tn': tn,
               'fp': fp,
               'fn': fn,
               'tp': tp,
               'acc': acc,
               'precision': precision,
               'recall': recall,
               'auc': roc_auc,
               'aps': aps,
               }
    df = pd.DataFrame(data=data_df, index=[r])
    df.to_csv(os.path.join(output, 'run_overviews', model_name+'_results_overview.csv'))

    data_df_tst = {'runID': str(r),
               'n_epochs': 0,
               'tn': tn_tst,
               'fp': fp_tst,
               'fn': fn_tst,
               'tp': tp_tst,
               'acc': acc_tst,
               'precision': precision_tst,
               'recall': recall_tst,
               'auc': roc_auc_tst,
               'aps': aps_tst,
               }
    df_tst = pd.DataFrame(data=data_df_tst, index=[r])
    df_tst.to_csv(os.path.join(output, 'run_overviews', model_name+'_results_overview_tst.csv'))
