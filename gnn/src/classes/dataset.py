import pandas as pd
from sklearn.model_selection import train_test_split

from gnn.environment import Environment
from gnn.parameters import Parameters


class Dataset():

    @property
    def purchases_to_predict(self) -> pd.DataFrame:
        """The edges that we want to predict."""
        return type(self)._purchases_to_predict

    @property
    def old_purchases(self) -> pd.DataFrame:
        """DataFrame containing all the purchases history."""
        return type(self)._old_purchases

    @property
    def customers(self) -> pd.DataFrame:
        """Features of the customers."""
        return type(self)._customers

    @property
    def articles(self) -> pd.DataFrame:
        """Features of the articles."""
        return type(self)._articles

    @property
    def purchased_list(self) -> pd.DataFrame:
        """All the purchases we want to predict: likely to compare easily with the prediction and calculate the score."""
        return type(self)._purchases_to_predict

    @property
    def train_set_length(self) -> int:
        """Number of edges in the train set."""
        return type(self)._train_set_length

    @property
    def valid_set_length(self) -> int:
        """Number of edges in the validation set."""
        return type(self)._valid_set_length

    @property
    def test_set_length(self) -> int:
        """Number of edges in the test set."""
        return type(self)._test_set_length

    def __init__(self, environment: Environment, parameters: Parameters):
        """Loads and prepare all the elements of the Dataset.

        Args:
            environment (Environment): all environment variables.
            parameters (Parameters): all parameters of the model.
        """

        # Load transactions dataset.
        transactions = pd.read_pickle(environment.transactions_path)

        self._purchases_to_predict = transactions[transactions['week_number'] == 0]
        self._old_purchases = transactions[transactions['week_number']
                                           <= parameters['weeks_of_purchases']]

        customer_id_list = self._old_purchases['customer_id'].unique()
        article_id_list = self._old_purchases['article_id'].unique()

        self._old_purchases = self._old_purchases[self._old_purchases['week_number'] > 0]

        del transactions

        # Load customer and article features.
        self._customers = pd.read_pickle(
            'pickles/self._customers_second_iteration.pkl')
        self._articles = pd.read_pickle(
            'pickles/self._articles_second_iteration.pkl')

        # Split self._customers into train / valid / test set through a field `set` with
        # values 0 / 1 / 2.
        customer_id_train, customer_id_test = train_test_split(
            self._purchases_to_predict.customer_id.unique(), test_size=parameters['test_size'])
        customer_id_train, customer_id_valid = train_test_split(
            customer_id_train, test_size=parameters['valid_size'])

        test_customers = pd.DataFrame(
            customer_id_test, columns=['customer_id'])
        test_customers['set'] = 2

        valid_customers = pd.DataFrame(
            customer_id_valid, columns=['customer_id'])
        valid_customers['set'] = 1

        self._customers = self._customers.merge(
            test_customers, on='customer_id', how='left')

        self._customers = self._customers.merge(
            valid_customers, on='customer_id', how='left')

        self._customers['set'].fillna(0, inplace=True)

        # Only process  customers who have transactions.
        customer_id = pd.Series(customer_id_list).rename('customer_id')
        self._customers = customer_id.merge(
            self._customers, on='customer_id', how='right')

        # Only process concerned articles who have transactions.
        article_id = pd.Series(article_id_list).rename('article_id')
        self._articles = article_id.merge(
            self._articles, on='article_id', how='right')

        # Change indexes types in order to save memory.
        self._articles = self._articles.reset_index().rename(
            {"index": "article_nid"}, axis=1)
        self._customers = self._customers.reset_index().rename(
            {"index": "customer_nid"}, axis=1)

        self._articles['article_nid'] = self._articles['article_nid'].astype(
            'int32')
        self._customers['customer_nid'] = self._customers['customer_nid'].astype(
            'int32')

        # Update transaction lists with new IDs and validation set Tag.
        self._old_purchases = self._old_purchases.merge(
            self._articles[['article_id', 'article_nid']], on='article_id', how='left')

        self._old_purchases = self._old_purchases.merge(
            self._customers[['customer_id', 'customer_nid', 'set']], on='customer_id', how='left')

        self._purchases_to_predict = self._purchases_to_predict.merge(
            self._articles[['article_id', 'article_nid']], on='article_id', how='left')

        self._purchases_to_predict = self._purchases_to_predict.merge(
            self._customers[['customer_id', 'customer_nid', 'set']], on='customer_id', how='left')

        self._purchases_to_predict['set'].fillna(0, inplace=True)

        self._purchases_to_predict = self._purchases_to_predict[[
            'article_nid', 'customer_nid', 'set']]

        # Add a graph specific ID for purchases to predict.
        self._purchases_to_predict = self._purchases_to_predict.reset_index()  \
            .rename({'index': 'eid'}, axis=1)

        self._purchases_to_predict['eid'] = self._purchases_to_predict['eid'].astype(
            'int32')

        # Get a grouped version of the purchase list.
        self._purchased_list = self._purchases_to_predict.groupby(
            ['customer_nid'], as_index=False).agg(
            articles=(
                'article_nid', lambda x: list(x)))

        self._purchased_list['length'] = self._purchased_list['articles'].apply(
            lambda x: len(x))

        # Get informations about the dataset.
        self._train_set_length = len(
            self._purchases_to_predict[self._purchases_to_predict['set'] == 0])
        self._valid_set_length = len(
            self._purchases_to_predict[self._purchases_to_predict['set'] == 1])

        # Group the old purchases in order to have one edge per (article,
        # customer), with some extra columns.
        self._old_purchases = self._old_purchases.groupby(
            ['customer_nid', 'article_nid'], as_index=False
        ).agg(
            sales_channel_id=('sales_channel_id', 'mean'),
            first=('day_number', 'min'),
            last=('day_number', 'max'),
            purchases=('article_nid', 'count'),
        )

        self._old_purchases['mean_interval'] = (
            (self._old_purchases['last'] - self._old_purchases['first'])
            / self._old_purchases['purchases']
        ).astype("int32")
