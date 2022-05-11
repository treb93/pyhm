import math
from typing import Tuple
import dgl
import pandas as pd
import numpy as np
import torch as th
from dgl.dataloading import DataLoader
from environment import Environment
from src.classes.graphs import Graphs
from parameters import Parameters
from src.classes.dataset import Dataset


class DataLoaders():
    def __init__(
            self,
            graphs: Graphs,
            dataset: Dataset,
            parameters: Parameters,
            environment: Environment):
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
        if not parameters.embedding_layer:
            n_layers = n_layers + 1

        # TODO: Update hardcoded numbers with params
        #if parameters.neighbor_sampling:
        #    edge_sampler = dgl.dataloading.NeighborSampler([parameters.neighbor_sampling for i in range(n_layers )])
        #else:
        edge_sampler = dgl.dataloading.NeighborSampler([0])
            
        node_sampler = dgl.dataloading.NeighborSampler([*[2 for i in range(n_layers - 1)], 2])

        # Batch size parameter corresponds to the positive edges, whereas our parameter corresponds to the total batch size. 
        edge_pos_size = parameters.edge_batch_size // (parameters.neg_sample_size + 1)

        print("Batch size: ", edge_pos_size)

        negative_sampler = dgl.dataloading.as_edge_prediction_sampler(
            edge_sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(
                parameters.neg_sample_size))

        self._dataloader_train_loss = dgl.dataloading.DataLoader(
            # graphs.full_graph if parameters.neighbor_sampling else 
            graphs.prediction_graph,
            {
                'buys': th.tensor(dataset.purchases_to_predict.loc[dataset.purchases_to_predict['set'] == 0].index.values, dtype=th.int32)
            },
            negative_sampler,
            batch_size=edge_pos_size,
            #device = environment.device,
            #use_uva = True,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        self._dataloader_valid_loss = dgl.dataloading.DataLoader(
            # graphs.full_graph if parameters.neighbor_sampling else 
            graphs.prediction_graph,
            {
                'buys': th.tensor(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 1].index.values, dtype=th.int32)
            },
            negative_sampler,
            batch_size=edge_pos_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        self._dataloader_embedding = dgl.dataloading.DataLoader(
            graphs.history_graph,
            {
                'customer': th.tensor(dataset.purchase_history['customer_nid'].unique(), dtype=th.int32),
                'article': th.tensor(dataset.purchase_history['article_nid'].unique(), dtype=th.int32)
            },
            node_sampler,
            batch_size=50000,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        self._num_batches_train = math.ceil(
            len(dataset.purchases_to_predict.loc[dataset.purchases_to_predict['set'] == 0]) /
            edge_pos_size)

        self._num_batches_valid = math.ceil(
            len(dataset.purchases_to_predict[dataset.purchases_to_predict['set'] == 1]) / edge_pos_size)

    @property
    def dataloader_train_loss(self) -> DataLoader:
        """Positive & negative edges for training."""
        return self._dataloader_train_loss
    
    @property
    def dataloader_valid_loss(self) -> DataLoader:
        """Positive & negative edges for loss calculation on validation phase."""
        return self._dataloader_valid_loss

    @property
    def dataloader_embedding(self) -> DataLoader:
        """Batches of customers and articles for embedding calculation on whole dataset."""
        return self._dataloader_embedding

    @property
    def num_batches_train(self) -> int:
        """Number of training batches."""
        return self._num_batches_train

    @property
    def num_batches_valid(self) -> DataLoader:
        """Number of validation batches."""
        return self._num_batches_valid
