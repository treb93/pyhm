import math
from typing import Tuple
import dgl
import pandas as pd
import numpy as np
import torch as th
from dgl.dataloading import DataLoader
from parameters import Parameters
from src.classes.dataset import Dataset


class DataLoaders():
    def __init__(
            self,
            graph: dgl.DGLHeteroGraph,
            dataset: Dataset,
            parameters: Parameters,
            environment):
        """
        Since data is large, it is fed to the model in batches. This creates batches for train, valid & test.

        Process:
            - Set up
                - Fix the number of layers. If there is an explicit embedding layer, we need 1 less layer in the blocks.
                - The sampler will generate computation blocks. Currently, only 'full' sampler is used, meaning that all
                nodes have all their neighbors, but one could specify 'partial' neighborhood to have only message passing
                with a limited number of neighbors.
                - The negative sampler generates K negative samples for all positive examples in the batch.
            - DataLoader : we use DataLoader function with negative sampler for generating positive / negative examples among 'will-buy' edges.
            During the training we iterate through these dataloaders in order to generate batches.

        Returns
            - dataloader_train          (dgl.dataloading.DataLoader) : Positive and negative links to train with.
            - dataloader_valid_loss     (dgl.dataloading.DataLoader) : Positive and negative links to use for validation loss calculation.
            - dataloader_valid_metrics  (dgl.dataloading.DataLoader) : Customer and articles needed for validation scoring.
            - dataloader_test           (dgl.dataloading.DataLoader) : Customer and articles needed for test scoring.
        """

        # Define the number of layers depending on the model's structure.
        n_layers = parameters.n_layers
        if parameters.embedding_layer:
            n_layers = n_layers - 1

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

        negative_sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(
                parameters.neg_sample_size))

        self._dataloader_train_loss = dgl.dataloading.DataLoader(
            graph,
            {
                'will-buy': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 0].index.values, dtype=th.int32).to(environment.device)
            },
            negative_sampler,
            batch_size=parameters.batch_size,
            device = environment.device,
            use_uva = True,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0)

        self._dataloader_train_metrics = dgl.dataloading.DataLoader(
            graph,
            {
                'customer': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 0]['customer_nid'].unique()).to(environment.device),
                'article': th.tensor(dataset.purchases_to_predict['article_nid'].unique()).to(environment.device)
            },
            sampler,
            batch_size=parameters.batch_size,
            device = environment.device,
            use_uva = True,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        self._dataloader_valid_loss = dgl.dataloading.DataLoader(
            graph,
            {
                'will-buy': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 1].index.values, dtype=th.int32).to(environment.device)
            },
            negative_sampler,
            batch_size=parameters.batch_size,
            device = environment.device,
            use_uva = True,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=0
        )

        self._dataloader_valid_metrics = dgl.dataloading.DataLoader(
            graph,
            {
                'customer': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 1]['customer_nid'].unique()).to(environment.device),
                'article': th.tensor(dataset.purchases_to_predict['article_nid'].unique()).to(environment.device)
            },
            sampler,
            batch_size=parameters.batch_size,
            device = environment.device,
            use_uva = True,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        self._dataloader_test = dgl.dataloading.DataLoader(
            graph,
            {
                'customer': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 2]['customer_nid'].unique()),
                'article': th.tensor(dataset.purchases_to_predict['article_nid'].unique())
            },
            sampler,
            batch_size=parameters.batch_size,
            device = environment.device,
            use_uva = True,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        self._num_batches_train = math.ceil(
            dataset.train_set_length /
            parameters.batch_size)

        self._num_batches_valid = math.ceil(
            dataset.valid_set_length / parameters.batch_size)

    @property
    def dataloader_train_loss(self) -> DataLoader:
        """Positive & negative edges for training."""
        return self._dataloader_train_loss

    @property
    def dataloader_train_metrics(self) -> DataLoader:
        """Batches of customers for metrics calculation, as we need to do it on a whole purchase list basis."""
        return self._dataloader_train_loss

    @property
    def dataloader_valid_loss(self) -> DataLoader:
        """Positive & negative edges for loss calculation."""
        return self._dataloader_train_loss

    @property
    def dataloader_valid_metrics(self) -> DataLoader:
        """Batches of customers for metrics calculation, as we need to do it on a whole purchase list basis."""
        return self._dataloader_train_loss

    @property
    def dataloader_test(self) -> DataLoader:
        """Batches of customers for metrics calculation, as we need to do it on a whole purchase list basis."""
        return self._dataloader_test

    @property
    def num_batches_train(self) -> int:
        """Number of training batches."""
        return self._num_batches_train

    @property
    def num_batches_valid(self) -> DataLoader:
        """Number of validation batches."""
        return self._num_batches_valid
