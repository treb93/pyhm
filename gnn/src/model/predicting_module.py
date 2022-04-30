
import torch
import torch.nn as nn


class PredictingModule(nn.Module):
    """
    Predicting module that incorporate the predicting layer defined earlier.

    Only used if fixed_params.pred == 'nn'.
    Process:
        - Fetches hidden states of 'heads' and 'tails' of the edges.
        - Concatenates them, then passes them through the predicting layer.
        - Returns ratings (sigmoid function).
    """

    def __init__(self, predicting_layer, embed_dim: int):
        super(PredictingModule, self).__init__()
        self.layer_nn = predicting_layer(embed_dim)

    def forward(self,
                graph,
                h
                ):
        ratings_dict = {}
        for etype in graph.canonical_etypes:
            if etype[0] in [
                    'customer',
                    'article'] and etype[2] in [
                    'customer',
                    'article']:
                utype, _, vtype = etype
                src_nid, dst_nid = graph.all_edges(etype=etype)
                emb_heads = h[utype][src_nid]
                emb_tails = h[vtype][dst_nid]
                cat_embed = torch.cat((emb_heads, emb_tails), 1)
                ratings = self.layer_nn(cat_embed)
                ratings_dict[etype] = torch.flatten(ratings)
        ratings_dict = {
            key: torch.unsqueeze(
                ratings_dict[key],
                1) for key in ratings_dict.keys()}
        return ratings_dict
