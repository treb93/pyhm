

import torch.nn as nn
import torch.nn.functional as F


class PredictingLayer(nn.Module):
    """
    Scoring function that uses a neural network to compute similarity between user and item.

    Only used if fixed_params.pred == 'nn'.
    Given the concatenated hidden states of heads and tails vectors, passes them through neural network and
    returns sigmoid ratings.
    """

    def reset_parameters(self):
        gain_relu = nn.init.calculate_gain('relu')
        gain_sigmoid = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.hidden_1.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.hidden_2.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.output.weight, gain=gain_sigmoid)

    def __init__(self, embed_dim: int):
        super(PredictingLayer, self).__init__()
        self.hidden_1 = nn.Linear(embed_dim * 2, 128)
        self.hidden_2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
