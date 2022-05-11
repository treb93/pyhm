from src.classes.graphs import Graphs
from parameters import Parameters

def get_dimension_dictionnary(
    graphs: Graphs,
    parameters: Parameters
):
    return {'customer': graphs.history_graph.nodes['customer'].data['features'].shape[1],
                'article': graphs.history_graph.nodes['article'].data['features'].shape[1],
                'edge': graphs.history_graph.edges['buys'].data['features'].shape[1],
                'out': parameters.out_dim,
                'hidden': parameters.hidden_dim}