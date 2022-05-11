
from pytest import param
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn

from parameters import Parameters
from src.model.conv_layer import ConvLayer
from src.model.cosine_prediction import CosinePrediction
from src.model.node_embedding import NodeEmbedding
from src.model.predicting_layer import PredictingLayer
from src.model.predicting_module import PredictingModule
from src.model.pass_through import PassThrough


class ConvModel(nn.Module):
    """
    Assembles embedding layers, multiple ConvLayers and chosen predicting function into a full model.

    """

    def __init__(self,
                 dimension_dictionnary,
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
        self.neighbor_sampling = parameters.neighbor_sampling

        if parameters.embedding_layer:
            self.user_embed = NodeEmbedding(
                dimension_dictionnary['customer'], dimension_dictionnary['hidden'])
            self.item_embed = NodeEmbedding(
                dimension_dictionnary['article'], dimension_dictionnary['hidden'])

        self.layers = nn.ModuleList()

        # If we use neighbor sampling, links to predict will be injected to the Conv layer so we need to handle and bypass it.
        # if self.neighbor_sampling:
        #     pass_through_layer = {
        #         'will-buy': PassThrough(dimension_dictionnary['hidden'], parameters)
        #     }
        #     
        #     pass_through_out_layer = {
        #         'will-buy': PassThrough(dimension_dictionnary['out'], parameters)
        #     }
        # else :
        # pass_through_layer = {}
        # pass_through_out_layer = {}

        # input layer
        if not parameters.embedding_layer:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        'buys': ConvLayer((dimension_dictionnary['customer'], dimension_dictionnary['article'], dimension_dictionnary['edge']), dimension_dictionnary['hidden'], parameters),
                        'is-bought-by': ConvLayer((dimension_dictionnary['article'], dimension_dictionnary['customer'], dimension_dictionnary['edge']), dimension_dictionnary['hidden'], parameters),
                        #**pass_through_layer
                    },
                    aggregate=parameters.aggregator_hetero
                )
            )

        # hidden layers
        for i in range(parameters.n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        'buys': ConvLayer((dimension_dictionnary['hidden'], dimension_dictionnary['hidden'], dimension_dictionnary['edge']), dimension_dictionnary['hidden'], parameters),
                        'is-bought-by': ConvLayer((dimension_dictionnary['hidden'], dimension_dictionnary['hidden'], dimension_dictionnary['edge']), dimension_dictionnary['hidden'], parameters),
                        #**pass_through_layer
                    },
                    aggregate=parameters.aggregator_hetero
                )
            )

        # output layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                        'buys': ConvLayer((dimension_dictionnary['hidden'], dimension_dictionnary['hidden'], dimension_dictionnary['edge']), dimension_dictionnary['out'], parameters),
                        'is-bought-by': ConvLayer((dimension_dictionnary['hidden'], dimension_dictionnary['hidden'], dimension_dictionnary['edge']), dimension_dictionnary['out'], parameters),
                        #**pass_through_out_layer
                },
                aggregate=parameters.aggregator_hetero
                )
            )

        if parameters.prediction_layer == 'cos':
            self.prediction_fn = CosinePrediction()
        elif parameters.prediction_layer == 'nn':
            self.prediction_fn = PredictingModule(PredictingLayer, dimension_dictionnary['out'])
        else:
            raise KeyError(
                'Prediction function {} not recognized.'.format(
                    parameters.pred))

    def get_embeddings(self,
                       graph_or_blocks,
                       h):
        
        if self.embedding_layer:
            h['customer'] = self.user_embed(h['customer'])
            h['article'] = self.item_embed(h['article'])
        else :
            h['customer'] = h['customer'].to(torch.float32)
            h['article'] = h['article'].to(torch.float32)
            
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            # Check wether graph is a block list or a full graph. 
            if type(graph_or_blocks) is list:
                h = layer(graph_or_blocks[i], h)
            else:
                h = layer(graph_or_blocks, h)
    
        return h

    def forward(self,
                pos_g,
                neg_g,
                embeddings
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
        pos_g:
            Positive graph, generated by the EdgeDataLoader. Contains all positive examples of the batch that need to
            be scored.
        neg_g:
            Negative graph, generated by the EdgeDataLoader. For all positive pairs in the pos_g, multiple negative
            pairs were generated. They are all scored.

        Returns
        -------
        pos_score:
            All scores between positive examples (aka positive pairs).
        neg_score:
            All score between negative examples.

        """
        
        pos_g.nodes['article'].data['h'] = embeddings['article'][pos_g.nodes['article'].data['_ID'].long()]
        pos_g.nodes['customer'].data['h'] = embeddings['customer'][pos_g.nodes['customer'].data['_ID'].long()]
        
        neg_g.nodes['article'].data['h'] = embeddings['article'][neg_g.nodes['article'].data['_ID'].long()]
        neg_g.nodes['customer'].data['h'] = embeddings['customer'][neg_g.nodes['customer'].data['_ID'].long()]

        pos_score = self.prediction_fn(pos_g)
        neg_score = self.prediction_fn(neg_g)
        
        return pos_score, neg_score
