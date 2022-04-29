import dgl
import numpy as np
import torch as th


def generate_dataloaders(graph,
                         purchases_to_predict,
                         # train_graph,
                         # train_eids_dict,
                         # valid_eids_dict,
                         # subtrain_uids,
                         # valid_uids,
                         # test_uids,
                         # all_iids,
                         # fixed_params,
                         # num_workers,
                         # all_sids=None,
                         #embedding_layer: bool = True,
                         parameters,
                         ):
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
        - dataloader_train (dgl.dataloading.DataLoader)
        - dataloader_valid (dgl.dataloading.DataLoader)
        - dataloader_test  (dgl.dataloading.DataLoader)
    """

    # Define the number of layers depending on the model's structure.
    n_layers = parameters.n_layers
    if parameters.embedding_layer:
        n_layers = n_layers - 1

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

    negative_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(
            parameters.neg_sample_size))

    dataloader_train = dgl.dataloading.DataLoader(
        graph,
        {
            'will-buy': th.tensor(purchases_to_predict[purchases_to_predict['set'] == 0].index.values, dtype=th.int32)
        },
        negative_sampler,
        batch_size=10,
        shuffle=True,
        drop_last=False,
        pin_memory=True)

    dataloader_valid = dgl.dataloading.DataLoader(
        graph,
        {
            'will-buy': th.tensor(purchases_to_predict[purchases_to_predict['set'] == 1].index.values, dtype=th.int32)
        },
        negative_sampler,
        batch_size=parameters.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=parameters.num_workers
    )

    dataloader_test = dgl.dataloading.DataLoader(
        graph,
        {
            'will-buy': th.tensor(purchases_to_predict[purchases_to_predict['set'] == 2].index.values, dtype=th.int32)
        },
        negative_sampler,
        batch_size=parameters.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=parameters.num_workers
    )

    return dataloader_train, dataloader_valid, dataloader_test
