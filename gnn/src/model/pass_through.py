from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn

from dgl.nn.pytorch import SAGEConv

from parameters import Parameters



class PassThrough(nn.Module):
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
        # nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)

    def __init__(self,
                 in_feats: Tuple[int, int],
                 out_feats: int,
                 parameters: Parameters
                 ):
        super().__init__()
        self._out_feats = out_feats
        
        if parameters.aggregator_type == 'lstm':
            self.lstm = nn.LSTM(
                self._in_neigh_feats,
                self._in_neigh_feats,
                batch_first=True)
        self.reset_parameters()

    def forward(self,
                graph: dgl.DGLHeteroGraph,
                h):
        

        return torch.zeros(h[1].shape[0], self._out_feats)
