import numpy as np
import pandas as pd
import torch
import sys

from src.builder import (create_ids, df_to_adjacency_list,
                         format_dfs, import_features)


class DataPaths:
    def __init__(self):
        self.result_filepath = 'outputs/results.txt'
        self.sport_feat_path = '../pickles/gnn_sports.pkl'
        self.train_path = '../pickles/gnn_user_item_complete.pkl'
        self.test_path = '../pickles/gnn_user_item_empty.pkl'
        self.item_sport_path = '../pickles/gnn_item_sport.pkl'
        self.user_sport_path = '../pickles/gnn_user_sport.pkl'
        self.sport_sportg_path = '../pickles/gnn_sport_groups.pkl'
        self.item_feat_path = '../pickles/gnn_item_features.pkl'
        self.user_feat_path = '../pickles/gnn_user_features.pkl'
        self.sport_onehot_path = '../pickles/gnn_sports.pkl'


def assign_graph_features(graph,
                          fixed_params,
                          data,
                          **params,
                          ):
    """
    Assigns features to graph nodes and edges, based on data previously provided in the dataloader.

    Parameters
    ----------
    graph:
        Graph of type dgl.DGLGraph, with all the nodes & edges.
    fixed_params:
        All fixed parameters. The only fixed params used are related to id types and occurrences.
    data:
        Object that contains node feature dataframes, ID mapping dataframes and user item interactions.
    params:
        Parameters used in this function include popularity & recency hyperparameters.

    Returns
    -------
    graph:
        The input graph but with features assigned to its nodes and edges.
    """
    # Assign features
    features_dict = import_features(
        graph,
        data.user_feat_df,
        data.item_feat_df,
        data.sport_onehot_df,
        data.ctm_id,
        data.pdt_id,
        data.spt_id,
        data.user_item_train,
        params['use_popularity'],
        params['days_popularity'],
        fixed_params.item_id_type,
        fixed_params.ctm_id_type,
        fixed_params.spt_id_type,
    )

    graph.nodes['user'].data['features'] = features_dict['user_feat']
    graph.nodes['item'].data['features'] = features_dict['item_feat']
    if 'sport' in graph.ntypes:
        graph.nodes['sport'].data['features'] = features_dict['sport_feat']

    # add date as edge feature
    if params['use_recency']:
        df = data.user_item_train_grouped
        df['max_date'] = max(df.hit_date)
        df['days_recency'] = (pd.to_datetime(
            df.max_date) - pd.to_datetime(df.hit_date)).dt.days + 1
        if fixed_params.discern_clicks:
            recency_tensor_buys = torch.tensor(
                df[df.buy == 1].days_recency.values)
            recency_tensor_clicks = torch.tensor(
                df[df.buy == 0].days_recency.values)
            graph.edges['buys'].data['recency'] = recency_tensor_buys
            graph.edges['bought-by'].data['recency'] = recency_tensor_buys
            graph.edges['clicks'].data['recency'] = recency_tensor_clicks
            graph.edges['clicked-by'].data['recency'] = recency_tensor_clicks
        else:
            recency_tensor = torch.tensor(df.days_recency.values)
            graph.edges['buys'].data['recency'] = recency_tensor
            graph.edges['bought-by'].data['recency'] = recency_tensor

    if params['use_popularity']:
        graph.nodes['item'].data['popularity'] = features_dict['item_pop']

    if fixed_params.duplicates == 'count_occurrence':
        if fixed_params.discern_clicks:
            graph.edges['clicks'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['clicks_num'])
            graph.edges['clicked-by'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['clicks_num'])
            graph.edges['buys'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['purchases_num'])
            graph.edges['bought-by'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['purchases_num'])
        else:
            graph.edges['buys'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['user_item_num'])
            graph.edges['bought-by'].data['occurrence'] = torch.tensor(
                data.adjacency_dict['user_item_num'])

    return graph
