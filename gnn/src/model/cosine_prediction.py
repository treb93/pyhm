
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn



class CosinePrediction(nn.Module):
    """
    Scoring function that uses cosine similarity to compute similarity between user and item.

    Only used if fixed_params.pred == 'cos'.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph):
        with graph.local_scope():
            for etype in graph.canonical_etypes:
                graph.nodes[etype[0]].data['norm_h'] = F.normalize(
                    graph.nodes[etype[0]].data['h'])
                graph.nodes[etype[2]].data['norm_h'] = F.normalize(
                    graph.nodes[etype[2]].data['h'])
                graph.apply_edges(
                    fn.u_dot_v(
                        'norm_h',
                        'norm_h',
                        'cos'),
                    etype=etype)
            ratings = graph.edata['cos']
        return ratings


