import pandas as pd
import numpy as np


def map12_from_series(purchased_articles, prediction, max_cutoff = 12):
    """Process the map12 score from the purchased articles and a prediction.

    Args:
        purchased_articles (pd.Series): the list of purchased articles for each customer
        prediction (pd.Series): the prediction (list format) for each customer.

    Returns:
        A pd.Series object containing the score for each customer.
    """

    return pd.concat([purchased_articles.rename('purchased_articles'), prediction.rename('prediction')], axis=1).apply(
        lambda x: np.sum(
            np.fromiter(
                (np.where(
                    x.prediction[cutoff] in x.purchased_articles,
                    np.sum(
                        np.fromiter(
                            (np.where(
                                x.prediction[position] in x.purchased_articles,
                                1 / (cutoff + 1),
                                0
                            ) for position in range(0, cutoff + 1)),
                            float
                        )
                    ),
                    0
                ) for cutoff in range(0, min(len(x.prediction), max_cutoff))),
                float)
        ) / min(len(x.purchased_articles), max_cutoff),
        axis=1)
