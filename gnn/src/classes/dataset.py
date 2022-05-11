import pandas as pd
import numpy as np
import gc

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
    
    @property
    def articles_in_prediction_nid(self) -> np.ndarray:
        """List of the article node IDs in the prediction graph."""
        return self._articles_in_prediction_nid
    
    @property
    def customers_in_prediction_nid(self) -> np.ndarray:
        """List of the customer node IDs in the prediction graph."""
        return self._customers_in_prediction_nid
    

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
        

        purchase_history = transactions[transactions['week_number']
                                        <= parameters.weeks_of_purchases]

        # Remove a defined amout of customers from the set.
        if parameters.remove:
            customer_to_keep_ids = purchases_to_predict['customer_id'].unique()
            
            np.random.shuffle(customer_to_keep_ids)
            nb_customers = int(customer_to_keep_ids.shape[0] * (1 - parameters.remove))
            customer_to_keep_ids = customer_to_keep_ids[0:nb_customers]
            
            purchases_to_predict = purchases_to_predict[purchases_to_predict['customer_id'].isin(customer_to_keep_ids)]
            purchase_history = purchase_history[purchase_history['customer_id'].isin(customer_to_keep_ids)]
            
            del customer_to_keep_ids
        
        customers_in_prediction_ids = purchases_to_predict['customer_id'].unique()
        article_in_prediction_ids = purchases_to_predict['article_id'].unique()
        

        if not parameters.include_last_week_in_history:
            purchase_history = purchase_history[purchase_history['week_number'] > 0]
        
        
        
        # Separate customers who are in the prediction DataFrame from the others.
        # TODO: At this point we don't use articles that are not in purchase_history_with_prediction, 
        # because the article list has to be the same when switchin full set mode on / off (otherwise the scores are waisted)
        purchase_history_without_prediction = purchase_history[~purchase_history['customer_id'].isin(customers_in_prediction_ids)]
        purchase_history_with_prediction = purchase_history[purchase_history['customer_id'].isin(customers_in_prediction_ids)]
            
        if parameters.embedding_on_full_set == False:
        
            # Remerge the two DataFrame after reducing the without prediction one, according to the ratio parameter.
            if parameters.customers_without_prediction_ratio:
                # Get the list of not-to-be-predicted customers to add.
                nb_customers_to_add = int(len(customers_in_prediction_ids) * parameters.customers_without_prediction_ratio)
                customers_without_prediction = purchase_history_without_prediction['customer_id'].unique()[0:nb_customers_to_add]
                
                # Filter the related dataframe.
                purchase_history_without_prediction = purchase_history_without_prediction[purchase_history_without_prediction['customer_id'].isin(customers_without_prediction)]
                
                # Merge with the dataframes with & without prediction
                purchase_history = pd.concat([purchase_history_with_prediction, purchase_history_without_prediction])
                del customers_without_prediction
                
            else: 
                purchase_history = purchase_history_with_prediction
            
        # Remove customers that don't have history & articles that are not in purchase_history_with_prediction (for ID management purpose)
        customers_in_history_ids = purchase_history_with_prediction['customer_id'].unique()
        articles_in_history_ids = purchase_history_with_prediction['article_id'].unique()
                
        purchases_to_predict = purchases_to_predict[purchases_to_predict['customer_id'].isin(customers_in_history_ids)]
        purchases_to_predict = purchases_to_predict[purchases_to_predict['article_id'].isin(articles_in_history_ids)]
        
        del customers_in_history_ids
        del articles_in_history_ids
        
        
        # Update article and customer's lists.
        article_id_list = pd.concat([purchases_to_predict['article_id'], purchase_history['article_id']]).unique()
        customer_id_list = pd.concat([purchases_to_predict['customer_id'], purchase_history['customer_id']]).unique()
        customers_in_prediction_ids = purchases_to_predict['customer_id'].unique()
        
        # Only process customers who have transactions.
        customer_id = pd.Series(customer_id_list).rename('customer_id')
        customers = customers.merge(
            customer_id, on='customer_id', how='right')

        # Only process articles for which there is transactions.
        article_id = pd.Series(article_id_list).rename('article_id')
        articles = articles.merge(
            article_id, on='article_id', how='right')
        
                
        # Split customers into train / valid / test set through a field `set` with
        # values 0 / 1.

        customer_id_train, customer_id_valid = train_test_split(
            customers_in_prediction_ids, 
            test_size=parameters.valid_size, 
            shuffle = True
        )
        
        valid_customers = pd.DataFrame(
            customer_id_valid, columns=['customer_id'])
        valid_customers['set'] = 1

        customers = customers.merge(
            valid_customers, on='customer_id', how='left')

        customers['set'] = customers['set'].fillna(0).astype('int32')

        del transactions

        
        # Truncate article features if needed.
        if parameters.reduce_article_features == True:
            articles = articles.iloc[:, :141]


        # Generate node ids.
        articles = articles.reset_index().rename(
            {"index": "article_nid"}, axis=1)
        customers = customers.reset_index().rename(
            {"index": "customer_nid"}, axis=1)

        articles['article_nid'] = articles['article_nid'].astype(
            'int32')
        customers['customer_nid'] = customers['customer_nid'].astype(
            'int32')
        
        
        # Update transaction lists with new IDs and validation set Tag.
        purchase_history = articles[['article_id', 'article_nid']].merge(
            purchase_history, on='article_id', how='inner')

        purchase_history = customers[['customer_id', 'customer_nid']].merge(
            purchase_history, on='customer_id', how='inner')


        purchases_to_predict = articles[['article_id', 'article_nid']].merge(
            purchases_to_predict, on='article_id', how='inner')

        purchases_to_predict = customers[['customer_id', 'customer_nid', 'set']].merge(
            purchases_to_predict, on='customer_id', how='inner')

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
        
        self._purchased_list = purchased_list
        self._customers_nid_train = customers[customers['customer_id'].isin(customer_id_train)]['customer_nid'].unique()
        self._customers_nid_valid = customers[customers['set'] == 1]['customer_nid'].unique()
    
        del purchased_list

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
        self._purchases_to_predict = purchases_to_predict
        self._articles = articles
        self._customers = customers
        
        
        self._articles_in_prediction_nid = purchases_to_predict.sort_values('article_nid')['article_nid'].unique()
        self._customers_in_prediction_nid = purchases_to_predict.sort_values('customer_nid')['customer_nid'].unique()

        
        ### DATA CHECKS - Ensure the nids of prediction graph and history graph will correspond (!)
        # Check that the article and customer prediction nids are similar to a np.arange from 0.
        assert (self.articles_in_prediction_nid == np.arange(0, len(self.articles_in_prediction_nid))).min() # (Checks if it's true for every value in the array)
        assert (self.customers_in_prediction_nid == np.arange(0, len(self.customers_in_prediction_nid))).min()
    
        # Checks that the article and customer prediction nids corresponds to the first nids of the history graph.
        articles_in_history_ids = purchase_history.sort_values('article_nid')['article_nid'].unique()
        customers_in_history_ids = purchase_history.sort_values('customer_nid')['customer_nid'].unique()
        
        assert (self.articles_in_prediction_nid  == articles_in_history_ids[0 : len(self.articles_in_prediction_nid)]).min()
        assert (self.customers_in_prediction_nid  == customers_in_history_ids[0 : len(self.customers_in_prediction_nid)]).min()
        
        # Checks that the article and customer prediction nids corresponds to the first nids of the customer & article DataFrame.
        assert (self.articles_in_prediction_nid == self.articles.iloc[0:len(self.articles_in_prediction_nid)]['article_nid'].values).min() # (Checks if it's true for every value in the array)
        assert (self.customers_in_prediction_nid == self.customers.iloc[0:len(self.customers_in_prediction_nid)]['customer_nid'].values).min()
        
        # Checks that the nids of the customer & articles tables are ordered and contiguous from 0.
        assert (self.customers['customer_nid'].values == np.arange(0, len(self.customers))).min()
        assert (self.articles['article_nid'].values == np.arange(0, len(self.articles))).min()
        
        
        print("Number of customer in train set : ", len(self._customers_nid_train))
        print("Number of customer in Valid set : ", len(self._customers_nid_valid))
        
        # Memory cleaning
        del purchase_history
        del purchases_to_predict
        del articles
        del customers
        
        gc.collect()