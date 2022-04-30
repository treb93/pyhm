
import dgl
import numpy as np
import pandas as pd
import torch as th

from src.classes.dataset import Dataset


def create_graph(dataset: Dataset
                 ) -> dgl.DGLHeteroGraph:
    """
    Create graph based on adjacency list.
    """
    graph_data = {
        (
            'customer',
            'buys',
            'article'): (
            th.tensor(
                dataset._old_purchases['customer_nid'].values,
                dtype=th.int32),
            th.tensor(
                dataset.old_purchases['article_nid'].values,
                dtype=th.int32),
        ),
        ('article',
         'is-bought-by',
         'customer'): (
            th.tensor(
                dataset.old_purchases['article_nid'].values,
                dtype=th.int32),
            th.tensor(
                dataset.old_purchases['customer_nid'].values,
                dtype=th.int32),
        ),
        ('customer',
         'will-buy',
         'article'): (
            th.tensor(
                dataset.purchases_to_predict['customer_nid'].values,
                dtype=th.int32),
            th.tensor(
                dataset.purchases_to_predict['article_nid'].values,
                dtype=th.int32),
        )}

    graph = dgl.heterograph(
        graph_data,
        device = th.device('cpu')
    )

    print('Version 3')
    # Add features.
    columns_to_drop = ['customer_id', 'customer_nid']
    graph.nodes['customer'].data['features'] = th.tensor(
        dataset.customers.drop(
            columns=columns_to_drop,
            axis=1).values,
        dtype=th.int32)

    columns_to_drop = ['article_id', 'article_nid']
    graph.nodes['article'].data['features'] = th.tensor(
        dataset.articles.drop(
            columns=columns_to_drop,
            axis=1).values,
        dtype=th.int32)

    columns_to_drop = [
        'customer_nid',
        'article_nid']
    
    graph.edges['buys'].data['features'] = th.tensor(
        dataset.old_purchases.drop(
            columns=columns_to_drop,
            axis=1).values,
        dtype=th.int32)
    # Also use purchase amount as weight, i.e multiplication of the source message passing.
    graph.edges['buys'].data['weight'] = th.tensor(dataset.old_purchases['purchases'].values, dtype=th.int32)
    
    graph.edges['is-bought-by'].data['features'] = th.tensor(
        dataset.old_purchases.drop(
            columns=columns_to_drop,
            axis=1).values,
        dtype=th.int32)
    # Also use purchase amount as weight, i.e multiplication of the source message passing.
    graph.edges['is-bought-by'].data['weight'] = th.tensor(dataset.old_purchases['purchases'].values, dtype=th.int32)


    return graph
