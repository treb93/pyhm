from environment import Environment
from parameters import Parameters
from src.classes.dataset import Dataset
from src.model.conv_model import ConvModel
from src.utils import softmax
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph


def create_ground_truth(users, items):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user actually bought.
    """
    ground_truth_arr = np.stack((np.asarray(users), np.asarray(items)), axis=1)
    ground_truth_dict = defaultdict(list)
    for key, val in ground_truth_arr:
        ground_truth_dict[key].append(val)
    return ground_truth_dict


def create_already_bought(g: DGLHeteroGraph, bought_eids, etype='buys'):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user already bought.
    """
    users_train, items_train = g.find_edges(bought_eids, etype=etype)
    already_bought_arr = np.stack(
        (np.asarray(users_train), np.asarray(items_train)), axis=1)
    already_bought_dict = defaultdict(list)
    for key, val in already_bought_arr:
        already_bought_dict[key].append(val)
    return already_bought_dict


def get_recommendation_tensor(y,
                              parameters: Parameters,
                              environment: Environment
                              ) -> torch.tensor:
    """
    Computes K recommendations for all users, given hidden states.

    returns recommandations(torch.tensor): A tensor of shape (customers, parameters.k)
    """

    if parameters.prediction_layer == 'cos':

        cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(environment.device)
        similarities = cos(
            y['customer'].reshape(-1, parameters.out_dim, 1),
            y['article'].reshape(1, parameters.out_dim, -1)
        )

        # Get the indexes of the best score
        recommandations = torch.argsort(
            similarities, dim=1, descending=True)[:, 0:12].int()

        # Replace the indexes by real article ids.
        #recommandations = node_ids['article'][recommandations[:]]

    #elif parameters.prediction_layer == 'nn':
    #    # TODO: Cette boucle coûte très cher. Voir si on peut les faire tous
    #    # d'un coup
    #    for user_id in node_ids['customer']:
    #        counter = counter + 1
    #        print(f"Customer n°{counter}")
    #        user_emb = y['customer'][user_id]
#
    #        user_emb_rpt = torch.cat(
    #            len(node_ids['article']) * [user_emb]).reshape(-1, parameters.embed_dim)
#
    #        cat_embed = torch.cat((user_emb_rpt, y['article']), 1)
    #        ratings = model.predict.layer_nn(cat_embed)
#
    #        ratings_formatted = ratings.cpu().detach(
    #        ).numpy().reshape(len(node_ids['article']),)
#
    #        order = node_ids['customer'][np.argsort(-ratings_formatted)]
#
    #        rec = order[:parameters.k]
    #        recommandations[user_id] = rec
    else:
        raise KeyError(
            f'Prediction function {parameters.prediction_layer} not recognized.')

    return recommandations


def precision_at_k(
        recommendations: torch.tensor,
        customer_nids: torch.tensor,
        dataset: Dataset) -> float:
    """
    Given the recommendations and the purchase list, computes precision, recall & coverage.
    """

    # Builds a dataset for having both purchase list and prediction list aside.
    score_dataframe = pd.concat([
        pd.Series(customer_nids).rename('customer_nid'),
        pd.Series(recommendations.tolist()).rename('prediction')
    ], axis=1)
    
    score_dataframe['prediction'] = score_dataframe['prediction'].fillna("").apply(list)

    score_dataframe = score_dataframe.merge(
        dataset._purchased_list, on='customer_nid', how='left')
    
    score_dataframe['prediction_length'] = score_dataframe['prediction'].apply(lambda x: len(x) if x else 0)

    score_dataframe = score_dataframe[(score_dataframe['prediction_length'] > 0) & (score_dataframe['length'] > 0)]

    precision = score_dataframe.apply(
        lambda x: np.sum(
                np.where(
                    x['prediction'][i] in x['articles'],
                    1,
                    0
                ) for i in range(int(min(len(x['prediction']), x['length']))
            )
        ) / min(len(x['prediction']), x['length']), axis=1)

    return precision.mean()


def get_metrics_at_k(model: ConvModel,
                     y,
                     node_ids,
                     dataset: Dataset,
                     parameters: Parameters
                     ) -> float:
    """
    Function combining all previous functions: create already_bought & ground_truth dict, get recs and compute metrics.
    """
    recommendations = get_recommendation_tensor(
        y,
        node_ids,
        model,
        parameters)

    precision = precision_at_k(
        recommendations, node_ids, dataset)

    return precision


def MRR_neg_edges(model,
                  blocks,
                  pos_g,
                  neg_g,
                  etype,
                  neg_sample_size):
    """
    (Currently not used.) Computes Mean Reciprocal Rank for the positive edge, out of all negative edges considered.

    Since it computes only on negative edges considered, it is an heuristic of the real MRR.
    However, if you put neg_sample_size as the total number of items, can be considered as MRR
    (because it will create as many edges as there are items).
    """
    input_features = blocks[0].srcdata['features']
    _, pos_score, neg_score = model(blocks,
                                    input_features,
                                    pos_g, neg_g,
                                    etype)
    neg_score = neg_score.reshape(-1, neg_sample_size)
    rankings = torch.sum(neg_score >= pos_score, dim=1) + 1
    return np.mean(1.0 / rankings.cpu().numpy())
