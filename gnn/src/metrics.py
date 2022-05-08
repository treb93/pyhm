from typing import List, Tuple
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


def get_recommendation_nids(y,
                              parameters: Parameters,
                              environment: Environment,
                              model: ConvModel,
                              cutoff = 12,
                              ) -> torch.tensor:
    """
    Computes K recommendations for all users, given hidden states.

    returns recommandations(torch.tensor): A tensor of shape (customers, parameters.k)
    """

    if parameters.prediction_layer == 'cos':

        cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(environment.device)
        similarities = cos(
            y['customer'].reshape(-1, parameters.out_dim, 1),
            torch.transpose(y['article'], 0, 1).reshape(1, parameters.out_dim, -1)
        )

        # Get the indexes of the best scores.
        recommandations = torch.argsort(
            similarities, dim=1, descending=True)[:, 0:cutoff].int().cpu().numpy()

        # Replace the indexes by real article ids.
        #recommandations = articles_nid[recommandations[:]]

    elif parameters.prediction_layer == 'nn':
    #    # TODO: Cette boucle coûte très cher. Voir si on peut les faire tous
    #    # d'un coup
        for customer_embedding in y['customer']:
    #        counter = counter + 1
    #        print(f"Customer n°{counter}")
    #        user_emb = y['customer'][user_id]
#

            user_emb_rpt = torch.cat(
                y['article'].shape[0] * [customer_embedding]).reshape(-1, parameters.embed_dim)

            cat_embed = torch.cat((user_emb_rpt, y['article']), 1)
            ratings = model.predict.layer_nn(cat_embed)

            ratings_formatted = ratings.cpu().detach().numpy().reshape(1, y['article'].shape[0])

            order = np.argsort(-ratings_formatted)[:, 0:cutoff]
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
        dataset: Dataset,
        parameters: Parameters):
    """
    Given the recommendations and the purchase list, computes precision at chosen cutoffs (defined in parameters).
    """

    # Builds a dataset for having both purchase list and prediction list aside.
    score_dataframe = pd.concat([
        pd.Series(customer_nids).rename('customer_nid'),
        pd.Series(recommendations.tolist()).rename('prediction')
    ], axis=1)
    
    #score_dataframe['prediction'] = score_dataframe['prediction'].fillna("").apply(list)

    score_dataframe = score_dataframe.merge(
        dataset.purchased_list, on='customer_nid', how='left')
    
    score_dataframe['prediction_length'] = score_dataframe['prediction'].apply(lambda x: len(x) if x else 0)

    score_dataframe = score_dataframe[(score_dataframe['prediction_length'] > 0) & (score_dataframe['length'] > 0)]

    precision_list = []

    for k in parameters.precision_cutoffs:
        precision = score_dataframe.apply(
            lambda x: np.sum(
                        np.where(
                            x['prediction'][i] in x['articles'],
                            1,
                            0
                        ) for i in range(min(k, x['prediction_length']))
                    ) / min(k, x['length'], x['prediction_length']),
            axis=1)
        
        precision_list.append(precision.mean())
        
    del score_dataframe
        
    return precision_list



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
