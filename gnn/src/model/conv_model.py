
import torch.nn as nn
import dgl.nn.pytorch as dglnn

from parameters import Parameters
from src.model.conv_layer import ConvLayer
from src.model.cosine_prediction import CosinePrediction
from src.model.node_embedding import NodeEmbedding
from src.model.predicting_layer import PredictingLayer
from src.model.predicting_module import PredictingModule


class ConvModel(nn.Module):
    """
    Assembles embedding layers, multiple ConvLayers and chosen predicting function into a full model.

    """

    def __init__(self,
                 g,
                 dim_dict,
                 parameters: Parameters
                 ):
        """
        Initialize the ConvModel.

        Parameters
        ----------
        g:
            Graph, only used to query graph metastructure (fetch node types and edge types).
        n_layers:
            Number of ConvLayer.
        dim_dict:
            Dictionary with dimension for all input nodes, hidden dimension (aka embedding dimension), output dimension.
        norm, dropout, aggregator_type:
            See ConvLayer for details.
        aggregator_hetero:
            Since we are working with heterogeneous graph, all nodes will have messages coming from different types of
            nodes. However, the neighborhood messages are specific to a node type. Thus, we have to aggregate
            neighborhood messages from different edge types.
            Choices are 'mean', 'sum', 'max'.
        embedding_layer:
            Some GNN papers explicitly define an embedding layer, whereas other papers consider the first ConvLayer
            as the "embedding" layer. If true, an explicit embedding layer will be defined (using NodeEmbedding). If
            false, the first ConvLayer will have input dimensions equal to node features.

        """
        super().__init__()
        self.embedding_layer = parameters.embedding_layer
        if parameters.embedding_layer:
            self.user_embed = NodeEmbedding(
                dim_dict['customer'], dim_dict['hidden'])
            self.item_embed = NodeEmbedding(
                dim_dict['article'], dim_dict['hidden'])

        self.layers = nn.ModuleList()


        # input layer
        if not parameters.embedding_layer:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        'buys': ConvLayer((dim_dict['customer'], dim_dict['article'] + dim_dict['edge']), dim_dict['hidden'], parameters),
                        'is-bought-by': ConvLayer((dim_dict['article'], dim_dict['customer'] + dim_dict['edge']), dim_dict['hidden'], parameters)
                    },
                    aggregate=parameters.aggregator_hetero
                )
            )

        # hidden layers
        for i in range(parameters.n_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        'buys': ConvLayer((dim_dict['hidden'], dim_dict['hidden'] + dim_dict['edge']), dim_dict['hidden'], parameters),
                        'is-bought-by': ConvLayer((dim_dict['hidden'], dim_dict['hidden'] + dim_dict['edge']), dim_dict['hidden'], parameters)
                    },
                    aggregate=parameters.aggregator_hetero
                )
            )

        # output layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                        'buys': ConvLayer((dim_dict['hidden'], dim_dict['hidden'] + dim_dict['edge']), dim_dict['out'], parameters),
                        'is-bought-by': ConvLayer((dim_dict['hidden'], dim_dict['hidden'] + dim_dict['edge']), dim_dict['out'], parameters)
                },
                aggregate=parameters.aggregator_hetero
                )
            )

        if parameters.prediction_layer == 'cos':
            self.pred_fn = CosinePrediction()
        elif parameters.prediction_layer == 'nn':
            self.pred_fn = PredictingModule(PredictingLayer, dim_dict['out'])
        else:
            raise KeyError(
                'Prediction function {} not recognized.'.format(
                    parameters.pred))

    def get_embeddings(self,
                       blocks,
                       h):
        for i in range(len(blocks)):
            layer = self.layers[i]
            h = layer(blocks[i], h)
        return h

    def forward(self,
                blocks,
                h,
                pos_g,
                neg_g,
                embedding_layer: bool = True,
                ):
        """
        Full pass through the ConvModel.

        Process:
            - Embedding layer
            - get_repr: As many ConvLayer as wished. All "Layers" are composed of as many ConvLayer as there
                        are edge types.
            - Predicting layer predicts score for all positive examples and all negative examples.

        Parameters
        ----------
        blocks:
            Computational blocks. Can be thought of as subgraphs. There are as many blocks as there are layers.
        h:
            Initial state of all nodes.
        pos_g:
            Positive graph, generated by the EdgeDataLoader. Contains all positive examples of the batch that need to
            be scored.
        neg_g:
            Negative graph, generated by the EdgeDataLoader. For all positive pairs in the pos_g, multiple negative
            pairs were generated. They are all scored.

        Returns
        -------
        h:
            Updated state of all nodes
        pos_score:
            All scores between positive examples (aka positive pairs).
        neg_score:
            All score between negative examples.

        """
        if embedding_layer:
            h['customer'] = self.user_embed(h['customer'])
            h['article'] = self.item_embed(h['article'])
        h = self.get_embeddings(blocks, h)
        pos_score = self.pred_fn(pos_g, h)
        neg_score = self.pred_fn(neg_g, h)
        return h, pos_score, neg_score
