
import dgl
from inflection import parameterize
import numpy as np
import pandas as pd
import torch as th
from parameters import Parameters

from src.classes.dataset import Dataset

class Graphs():

    @property
    def history_graph(self) -> dgl.DGLHeteroGraph:
        """Graph with all purchase history, for embedding."""
        return self._history_graph

    @property
    def prediction_graph(self) -> dgl.DGLHeteroGraph:
        """Graph with all purchases to predict, using link prediction."""
        return self._prediction_graph

    @property
    def full_graph(self) -> dgl.DGLHeteroGraph:
        """Graph with all purchase data, used for both embedding and link prediction."""
        return self._full_graph


    def __init__(self, dataset: Dataset, parameters: Parameters
                    ):
        """
        Create two graphes, one for embedding and one for link prediction.
        """
        history_data = {
            (
                'customer',
                'buys',
                'article'): (
                th.tensor(
                    dataset.purchase_history['customer_nid'].values,
                    dtype=th.int32),
                th.tensor(
                    dataset.purchase_history['article_nid'].values,
                    dtype=th.int32),
            ),
            ('article',
            'is-bought-by',
            'customer'): (
                th.tensor(
                    dataset.purchase_history['article_nid'].values,
                    dtype=th.int32),
                th.tensor(
                    dataset.purchase_history['customer_nid'].values,
                    dtype=th.int32),
            )
            
        }
        
 
        
        # If we use neighbor sampling, we need to do embedding and link sampling on the same graph.
        history_graph = dgl.heterograph(
            #prediction_data if parameters.neighbor_sampling else 
            history_data,
        )
        
        # Add features.
        columns_to_drop = ['customer_id', 'customer_nid', 'set']
        history_graph.nodes['customer'].data['features'] = th.tensor(
            dataset.customers.drop(
                columns=columns_to_drop,
                axis=1).values)

        columns_to_drop = ['article_id', 'article_nid']
        history_graph.nodes['article'].data['features'] = th.tensor(
            dataset.articles.drop(
                columns=columns_to_drop,
                axis=1).values)

        columns_to_drop = [
            'customer_nid',
            'article_nid'
        ]
        
        history_graph.edges['buys'].data['features'] = th.tensor(
            dataset.purchase_history.drop(
                columns=columns_to_drop,
                axis=1).values, dtype=th.float32)
        # Also use purchase amount as weight, i.e multiplication of the source message passing.
        history_graph.edges['buys'].data['weight'] = th.tensor(dataset.purchase_history['purchases'].values, dtype=th.float32)
        
        history_graph.edges['is-bought-by'].data['features'] = th.tensor(
            dataset.purchase_history.drop(
                columns=columns_to_drop,
                axis=1).values, dtype=th.float32)
        # Also use purchase amount as weight, i.e multiplication of the source message passing.
        history_graph.edges['is-bought-by'].data['weight'] = th.tensor(dataset.purchase_history['purchases'].values, dtype=th.float32)
        
        
        # If we have to calculate recommandations on full set, create a placeholder for embeddings.
        if parameters.embedding_on_full_set:
            history_graph.nodes['customer'].data['h'] = th.zeros((history_graph.num_nodes('customer'), parameters.out_dim))
            history_graph.nodes['article'].data['h'] = th.zeros((history_graph.num_nodes('article'), parameters.out_dim))

        prediction_data = {
            ('customer',
             'buys',
             'article'): (
                th.tensor(
                    dataset.purchases_to_predict['customer_nid'].values,
                    dtype=th.int32),
                th.tensor(
                    dataset.purchases_to_predict['article_nid'].values,
                    dtype=th.int32),
            )
        }
        
        # If Neighbor Sampling is enabled, we use a single graph for embedding and prediction.
        # if parameters.neighbor_sampling:
        #     
        #     # Create a placeholder for embeddings.
        history_graph.nodes['customer'].data['h'] = th.zeros((history_graph.num_nodes('customer'), parameters.out_dim))
        history_graph.nodes['article'].data['h'] = th.zeros((history_graph.num_nodes('article'), parameters.out_dim))
        # 
        #     self._history_graph = history_graph
        #     self._full_graph = history_graph
        # else:
        prediction_graph = dgl.heterograph(
            prediction_data,
        )
        
        # Create a placeholder for embeddings.
        prediction_graph.nodes['customer'].data['h'] = th.zeros((prediction_graph.num_nodes('customer'), parameters.out_dim))
        prediction_graph.nodes['article'].data['h'] = th.zeros((prediction_graph.num_nodes('article'), parameters.out_dim))
        
        self._history_graph = history_graph
        self._prediction_graph = prediction_graph
       
        
        del history_graph
        del prediction_graph
    
        return
