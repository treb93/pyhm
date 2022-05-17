from sklearn.base import TransformerMixin
import pandas as pd
import swifter


class ListToUniclass(TransformerMixin):
    def __init__(self):
        return

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        self.list_for_dataframe = []

        self.total_rows = len(dataset)
        self.index = 0

        dataset.swifter.apply(lambda x: self.expand_row(x), axis=1)

        return pd.DataFrame(
            self.list_for_dataframe,
            columns=[
                "customer_id",
                "article_id",
                "label",
                "in_pair_list",
                "in_repurchase_list",
                'in_cross_list'])

    def expand_row(self, row):

        for article_id in row['shortlist']:
            self.list_for_dataframe.append([
                row['customer_id'],
                article_id,
                1 if article_id in row['purchase_list'] else 0,
                row['pair_list'].index(article_id) if article_id in row['pair_list'] else 100,
                row['repurchase_list'].index(article_id) if article_id in row['repurchase_list'] else 100,
                row['cross_list'].index(article_id) if article_id in row['cross_list'] else 100,
            ])

        self.index += 1
