from re import X
import pandas as pd
import numpy as np


def map12_from_score_table(score_table):
    """Process the map12 overall score from a by-article scoring.

    Args:
        Score table (pd.DataFrame): A DataFrame with columns 'customer_id', 'label', 'prediction'

    Returns:
        A pd.DataFrame object containing the score for each customer.
    """

    score_table.sort_values(['prediction'], ascending=False, inplace=True)

    score_list = score_table.groupby(
        'customer_id', as_index=False, sort=False
    ).agg(
        labels=(
            'label', lambda x: list(x)),
        predictions=(
            'prediction', lambda x: list(x)),
        nb_labels=('label', 'sum')
    )

    score_list['map12'] = score_list.apply(
        lambda x: np.sum(
            np.fromiter((
                np.where(
                    x.labels[cutoff] == 1,
                    np.sum(
                        np.fromiter(
                            (x.labels[position] / (cutoff + 1) for position in range(0, cutoff + 1)),
                            float
                        )
                    ),
                    0
                ) for cutoff in range(0, min(len(x.predictions), 12))
            ), float)
        ) / min(x.nb_labels, 12),
        axis=1)

    return score_list[['customer_id', 'map12']]
