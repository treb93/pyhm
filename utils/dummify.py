from sklearn.base import TransformerMixin
import pandas as pd


class Dummify(TransformerMixin):
    def __init__(self):
        return

    def fit(self):
        return self

    def transform(self, dataset):

        for column in dataset.columns:
            if not isinstance(dataset[column].dtype, pd.CategoricalDtype):
                continue

            dummies = pd.get_dummies(
                dataset[column], prefix=column, prefix_sep=":")
            dataset = pd.concat([dataset, dummies], axis=1)
            dataset.drop(columns=column, axis=1, inplace=True)

        return dataset

    def inverse_transform(self, dataset):

        columns = dataset.columns

        for column in columns:
            if ':' not in column:
                continue

            category_name = column.split(':')[0]
            label = column.split(':')[1]

            if category_name not in dataset.columns:
                dataset[category_name] = ''

                dataset[category_name] = dataset[[category_name, column]].apply(
                    lambda x: label if x[column] == 1 else x[category_name])

                dataset.drop(column=column, axis=1, inplace=True)
