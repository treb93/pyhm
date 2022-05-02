import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from environment import Environment
from parameters import Parameters


class Dataset():

    @property
    def purchases_to_predict(self) -> pd.DataFrame:
        """The edges that we want to predict."""
        return self._purchases_to_predict

    @property
    def purchase_history(self) -> pd.DataFrame:
        """DataFrame containing all the purchases history."""
        return self._purchase_history

    @property
    def customers(self) -> pd.DataFrame:
        """Features of the customers."""
        return self._customers

    @property
    def articles(self) -> pd.DataFrame:
        """Features of the articles."""
        return self._articles

    @property
    def purchased_list(self) -> pd.DataFrame:
        """All the purchases we want to predict: likely to compare easily with the prediction and calculate the score."""
        return self._purchased_list

    @property
    def customers_nid_train(self) -> np.ndarray:
        """List of the customer node IDs in the train set."""
        return self._customers_nid_train
    
    @property
    def customers_nid_valid(self) -> np.ndarray:
        """List of the customer node IDs in the valid set."""
        return self._customers_nid_valid
    

    def __init__(self, environment: Environment, parameters: Parameters):
        """Loads and prepare all the elements of the Dataset.

        Args:
            environment (Environment): all environment variables.
            parameters (Parameters): all parameters of the model.
        """

        # Load transactions dataset.
        transactions = pd.read_pickle(environment.transactions_path)
        
        purchases_to_predict = transactions[transactions['week_number'] == 0]
        
        
        # Load customer and article features.
        customers = pd.read_pickle(environment.customers_path)
        articles = pd.read_pickle(environment.articles_path)
        
        # Remove a defined amoutn of customers from the set.
        if parameters.remove:
            customers = customers.sample(frac = 1 - parameters.remove)
            customers_id_list = customers['customer_id'].unique()
            transactions = transactions[transactions['customer_id'].isin(customers_id_list)]

        if parameters.embedding_on_full_set == True:
            purchase_history = transactions[transactions['week_number']
                                            <= parameters.weeks_of_purchases]
        else:
            purchase_history = transactions[transactions['week_number']
                                            <= parameters.weeks_of_purchases]

            customer_id_list = purchases_to_predict['customer_id'].unique()
            article_id_list = purchases_to_predict['article_id'].unique()
            
            # Split customers into train / valid / test set through a field `set` with
            # values 0 / 1.

            customer_id_train, customer_id_valid = train_test_split(
            customer_id_list, test_size=parameters.valid_size)
            
            valid_customers = pd.DataFrame(
                customer_id_valid, columns=['customer_id'])
            valid_customers['set'] = 1

            customers = customers.merge(
                valid_customers, on='customer_id', how='left')

            customers['set'] = customers['set'].fillna(0).astype('int32')
            
            

            purchase_history = purchase_history[purchase_history['week_number'] > 0]
            
            # Separate customers who are in the prediction DataFrame from the others.
            purchase_history_with_prediction = purchase_history[purchase_history['customer_id'].isin(customer_id_list)]
            purchase_history_without_prediction = purchase_history[~purchase_history['customer_id'].isin(customer_id_list)]
            
            # Remerge the two DataFrame according to the related parameter.
            if parameters.customers_without_prediction_ratio:
                nb_customers_to_add = int(len(customer_id_list) * parameters.customers_without_prediction_ratio)
                customers_without_prediction = purchase_history_without_prediction['customer_id'].unique()[0:nb_customers_to_add]
                purchase_history_without_prediction = purchase_history_without_prediction[purchase_history_without_prediction['customer_id'].isin(customers_without_prediction)]
                purchase_history = pd.concat([purchase_history_with_prediction, purchase_history_without_prediction])
                del customers_without_prediction
                
            else: 
                purchase_history = purchase_history_with_prediction
                
            # Update article and customer's lists.
            customer_id_list = pd.concat([purchases_to_predict['customer_id'], purchase_history['customer_id']]).unique()
            article_id_list = pd.concat([purchases_to_predict['article_id'], purchase_history['article_id']]).unique()
            
            # Only process customers who have transactions.
            customer_id = pd.Series(customer_id_list).rename('customer_id')
            customers = customers.merge(
                customer_id, on='customer_id', how='right')

            # Only process articles for which there is transactions.
            article_id = pd.Series(article_id_list).rename('article_id')
            articles = articles.merge(
                article_id, on='article_id', how='right')

        del transactions




        # Change indexes types in order to save memory.
        articles = articles.reset_index().rename(
            {"index": "article_nid"}, axis=1)
        customers = customers.reset_index().rename(
            {"index": "customer_nid"}, axis=1)

        articles['article_nid'] = articles['article_nid'].astype(
            'int32')
        customers['customer_nid'] = customers['customer_nid'].astype(
            'int32')

        # Update transaction lists with new IDs and validation set Tag.
        purchase_history = purchase_history.merge(
            articles[['article_id', 'article_nid']], on='article_id', how='left')

        purchase_history = purchase_history.merge(
            customers[['customer_id', 'customer_nid', 'set']], on='customer_id', how='left')

        purchases_to_predict = purchases_to_predict.merge(
            articles[['article_id', 'article_nid']], on='article_id', how='left')

        purchases_to_predict = purchases_to_predict.merge(
            customers[['customer_id', 'customer_nid', 'set']], on='customer_id', how='left')

        purchases_to_predict['set'].fillna(0, inplace=True)

        purchases_to_predict = purchases_to_predict[[
            'article_nid', 'customer_nid', 'set']]

        # Add a graph specific ID for purchases to predict.
        purchases_to_predict = purchases_to_predict.reset_index()  \
            .rename({'index': 'eid'}, axis=1)

        purchases_to_predict['eid'] = purchases_to_predict['eid'].astype(
            'int32')

        # Get a grouped version of the purchase list.
        purchased_list = purchases_to_predict.groupby(
            ['customer_nid'], as_index=False).agg(
            articles=(
                'article_nid', lambda x: list(x)))

        purchased_list['length'] = purchased_list['articles'].apply(
            lambda x: len(x)).astype("int32")


        # Group the old purchases in order to have one edge per (article,
        # customer), with some extra columns.
        purchase_history['day_number'] = (purchase_history['t_dat'].max() - purchase_history['t_dat'] ).dt.days
        
        purchase_history = purchase_history.groupby(
            ['customer_nid', 'article_nid'], as_index=False
        ).agg(
            sales_channel_id=('sales_channel_id', 'mean'),
            first=('day_number', 'min'),
            last=('day_number', 'max'),
            purchases=('article_nid', 'count'),
        )

        purchase_history['mean_interval'] = (
            (purchase_history['last'] - purchase_history['first'])
            / purchase_history['purchases']
        ).astype("int32")

        self._purchase_history = purchase_history
        self._purchased_list = purchased_list
        self._purchases_to_predict = purchases_to_predict
        self._articles = articles
        self._customers = customers
        
        
        
        self._customers_nid_train = customers[customers['customer_id'].isin(customer_id_train)]['customer_nid'].unique()
        self._customers_nid_valid = customers[customers['set'] == 1]['customer_nid'].unique()
        
        
        print("Number of customer in train set : ", len(self._customers_nid_train))
        print("Number of customer in Valid set : ", len(self._customers_nid_valid))
        
        del purchase_history
        del purchased_list
        del purchases_to_predict
        del articles
        del customers
        