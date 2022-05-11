from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn

from dgl.nn.pytorch import SAGEConv

from parameters import Parameters



class ConvLayer(nn.Module):
    """
    1 layer of message passing & aggregation, specific to an edge type.

    Similar to SAGEConv layer in DGL library.

    Methods
    -------
    reset_parameters:
        Intialize parameters for all neural networks in the layer.
    _lstm_reducer:
        Aggregate messages of neighborhood nodes using LSTM technique. (All other message aggregation are builtin
        functions of DGL).
    forward:
        Actual message passing & aggregation, & update of nodes messages.

    """

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
        if self._aggre_type in [
            'pool_nn',
            'pool_nn_edge',
            'mean_nn',
                'mean_nn_edge']:
            nn.init.xavier_uniform_(self.fc_preagg.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()

    def __init__(self,
                 in_feats: Tuple[int, int, int],
                 out_feats: int,
                 parameters: Parameters
                 ):
        """
        Initialize the layer with parameters.

        Parameters
        ----------
        in_feats:
            Dimension of the message (aka features) of the node type neighbor and of the node type. E.g. if the
            ConvLayer is initialized for the edge type (user, clicks, item), in_feats should be
            (dimension_of_item_features, dimension_of_user_features). Note that usually features will first go
            through embedding layer, thus dimension might be equal to the embedding dimension.
        out_feats:
            Dimension that the output (aka the updated node message) should take. E.g. if the layer is a hidden layer,
            out_feats should be hidden_dimension, whereas if the layer is the output layer, out_feats should be
            output_dimension.
        dropout:
            Traditional dropout applied to input features.
        aggregator_type:
            This is the main parameter of ConvLayer; it defines how messages are passed and aggregated. Multiple
            choices:
                'mean' : messages are passed normally, and aggregated by doing the mean of all neighbor messages.
                'mean_nn' : messages are passed through a neural network before being passed to neighbors, and
                            aggregated by doing the mean of all neighbor messages.
                'pool_nn' : messages are passed through a neural network before being passed to neighbors, and
                            aggregated by doing the max of all neighbor messages.
                'lstm' : messages are passed normally, and aggregared using _lstm_reducer.
            All choices have also their equivalent that ends with _edge (e.g. mean_nn_edge). Those version include
            the edge in the message passing, i.e. the message will be multiplied by the value of the edge.
        norm:
            Apply normalization
        """
        super().__init__()
        self._in_neigh_feats, self._in_self_feats, self._in_edge_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = parameters.aggregator_type
        self.dropout_fn = nn.Dropout(parameters.dropout)
        self.norm = parameters.norm
        self.fc_self = nn.Linear(self._in_self_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_neigh_feats + self._in_edge_feats, out_feats, bias=False)
        # self.fc_edge = nn.Linear(1, 1, bias=True)  # Projecting recency days
        if parameters.aggregator_type in [
            'pool_nn',
            'pool_nn_weighted',
            'mean_nn',
                'mean_nn_weighted']:
            self.fc_preagg = nn.Linear(
                self._in_neigh_feats,
                self._in_neigh_feats,
                bias=False)
        if parameters.aggregator_type == 'lstm' or parameters.aggregator_type == 'lstm_weighted':
            self.lstm = nn.LSTM(
                self._in_neigh_feats,
                self._in_neigh_feats,
                batch_first=True)
        self.reset_parameters()

    def _lstm_reducer(self, nodes):
        """
        Aggregates the neighborhood messages using LSTM technique.

        This was taken from DGL docs. For computation optimization, when 'batching' nodes, all nodes
        with the same degrees are batched together, i.e. at first all nodes with 1 in-neighbor are computed
        (thus m.shape = [number of nodes in the batchs, 1, dimensionality of h]), then all nodes with 2 in-
        neighbors, etc.
        """
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_neigh_feats)),
             m.new_zeros((1, batch_size, self._in_neigh_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self,
                graph: dgl.DGLHeteroGraph,
                h):
        """
        Message passing & aggregation, & update of node messages.

        Process is the following:
            - Messages (h_neigh and h_self) are extracted from x
            - Dropout is applied
            - Messages are passed and aggregated using the _aggre_type specified (see __init__ for details), which
              return updated h_neigh
            - h_self passes through neural network & updated h_neigh passes through neural network, and are then summed
              up
            - The sum (z) passes through Relu
            - Normalization is applied
        """
        h_neigh, h_self = h
        h_neigh = self.dropout_fn(h_neigh)
        h_self = self.dropout_fn(h_self)
        
        graph.update_all(fn.copy_e('features', 'm_e'), fn.mean('m_e', 'edge'))
            
        if self._aggre_type == 'mean':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.copy_u('h', 'm_n'), 
                fn.mean('m_n', 'neigh')
            )
            #graph.update_all(fn.copy_u('h', 'm_n'), fn.mean('m_n', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)

        elif self._aggre_type == 'mean_weighted':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.u_mul_e('h', 'weight', 'm'), 
                fn.mean('m', 'neigh')
            )
            #graph.update_all(fn.copy_u('h', 'm_n'), fn.mean('m_n', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)
            
        elif self._aggre_type == 'mean_nn':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.copy_u('h', 'm'), 
                fn.mean('m', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)

        elif self._aggre_type == 'mean_nn_weighted':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.u_mul_e('h', 'weight', 'm'), 
                fn.mean('m', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)
            
        elif self._aggre_type == 'pool_nn':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.copy_u('h', 'm'), 
                fn.max('m', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)
            
        elif self._aggre_type == 'pool_nn_weighted':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.u_mul_e('h', 'weight', 'm'), 
                fn.max('m', 'neigh'))
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)

        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.copy_u('h', 'm'), 
                self._lstm_reducer)
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)
            
        elif self._aggre_type == 'lstm_weighted':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.u_mul_e('h', 'weight', 'm'), 
                self._lstm_reducer)
            h_neigh = torch.cat([graph.dstdata['neigh'], graph.dstdata['edge']], dim = 1)
           
        else:
            raise KeyError(
                'Aggregator type {} not recognized.'.format(
                    self._aggre_type))


        z = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        z = F.relu(z)

        # normalization
        if self.norm:
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0,
                                 torch.tensor(1.).to(z_norm),
                                 z_norm)
            z = z / z_norm

        # print(z.shape)

        return z
