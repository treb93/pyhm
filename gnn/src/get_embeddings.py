import torch
from gnn.environment import Environment
from gnn.parameters import Parameters

from gnn.src.model import ConvModel
from dgl.dataloading import DataLoader


def get_embeddings(graph,
                   trained_model: ConvModel,
                   dataloader: DataLoader,
                   num_batches,
                   environment: Environment,
                   parameters: Parameters):
    """
    Fetch the embeddings for all the nodes in the nodeloader.

    Nodeloader is preferable when computing embeddings because we can specify which nodes to compute the embedding for,
    and only have relevant nodes in the computational blocks. Whereas Edgeloader is preferable for training, because
    we generate negative edges also.

    Returns:
        y: The embeddings
        node_ids: A dictionnary containing lists of embedded nodes IDS for each type.
    """

    batch_index = 0

    node_ids = {
        ntype: [] for ntype in graph.ntypes
    }

    # Create placeholder for the embeddings.
    if environment.cuda:
        y = {
            ntype: torch.zeros(
                graph.num_nodes(ntype),
                parameters.out_dim
            ).to(environment.device) for ntype in graph.ntypes
        }
    else:
        y = {ntype: torch.zeros(graph.num_nodes(ntype), parameters.out_dim)
             for ntype in graph.ntypes}

    # Process all batches given by the Dataloader.
    for input_nodes, output_nodes, blocks in dataloader:
        batch_index += 1
        if batch_index % 10 == 0:
            print(
                "Computing embeddings: Batch {} out of {}".format(
                    batch_index, num_batches))

        if environment.cuda:
            blocks = [b.to(environment.device) for b in blocks]
        input_features = blocks[0].srcdata['features']

        if parameters.embedding_layer:
            input_features['user'] = trained_model.user_embed(
                input_features['user'])
            input_features['item'] = trained_model.item_embed(
                input_features['item'])

        h = trained_model.get_repr(blocks, input_features)

        for ntype in h.keys():
            y[ntype][output_nodes[ntype]] = h[ntype]
            node_ids[ntype].append(output_nodes[ntype])

    return y, node_ids
