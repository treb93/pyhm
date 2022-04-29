import torch
import torch.nn as nn


def max_margin_loss(pos_score,
                    neg_score,
                    parameters,
                    environment
                    ):
    """
    Simple max margin loss.

    Parameters
    ----------
    pos_score:
        All similarity scores for positive examples.
    neg_score:
        All similarity scores for negative examples.
    delta:
        Delta from which the pos_score should be higher than all its corresponding neg_score.
    neg_sample_size:
        Number of negative examples to generate for each positive example.
        See main.SearchableHyperparameters for more details.
    """
    all_scores = torch.empty(0)
    if environment.cuda:
        all_scores = all_scores.to(environment.device)
    for etype in pos_score.keys():
        neg_score_tensor = neg_score[etype]
        pos_score_tensor = pos_score[etype]
        neg_score_tensor = neg_score_tensor.reshape(
            -1, parameters.neg_sample_size)

        scores = neg_score_tensor + parameters.delta - pos_score_tensor
        relu = nn.ReLU()
        scores = relu(scores)

        all_scores = torch.cat((all_scores, scores), 0)
    return torch.mean(all_scores)
