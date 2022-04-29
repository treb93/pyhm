
import dgl
import numpy as np
import pandas as pd
import torch as th

from gnn.src.classes.dataset import Dataset


def create_graph(dataset: Dataset
                 ) -> dgl.DGLHeteroGraph:
    """
    Create graph based on adjacency list.
    """
    graph_data = {
        ('customer', 'buys', 'article'):
            (
                th.tensor(dataset.old_purchases['customer_nid'].values),
                th.tensor(dataset.old_purchases['article_nid'].values),
        ),

        ('article', 'is-bought-by', 'customer'):
            (
                th.tensor(dataset.old_purchases['article_nid'].values),
                th.tensor(dataset.old_purchases['customer_nid'].values),
        ),

        ('customer', 'will-buy', 'article'):
            (
                th.tensor(dataset.purchases_to_predict['customer_nid'].values),
                th.tensor(dataset.purchases_to_predict['article_nid'].values),
        )
    }

    graph = dgl.heterograph(
        graph_data
    )

    # Add features.
    columns_to_drop = ['customer_id', 'customer_nid']
    graph.nodes['customer'].data['features'] = th.tensor(
        dataset.customers.drop(columns=columns_to_drop, axis=1).values
    )

    columns_to_drop = ['article_id', 'article_nid']
    graph.nodes['articles'].data['features'] = th.tensor(
        dataset.customers.drop(columns=columns_to_drop, axis=1).values
    )

    graph.edges['buy']

    return graph
