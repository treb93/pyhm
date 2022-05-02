import torch
import torch.nn as nn

from environment import Environment
from parameters import Parameters


def max_margin_loss(pos_score: torch.tensor,
                    neg_score: torch.tensor,
                    parameters: Parameters,
                    environment: Environment
                    ) -> torch.Tensor:
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
    all_scores = torch.empty(0).to(environment.device)
        
    neg_score = neg_score.reshape(
        -1, parameters.neg_sample_size)

    scores = neg_score + parameters.delta - pos_score
    scores = scores.to(environment.device)
    
    relu = nn.ReLU()
    scores = relu(scores)
    
    all_scores = torch.cat((all_scores, scores), 0)
    return torch.mean(all_scores)
