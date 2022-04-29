import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset(environment, parameters):
    """_summary_

    Args:
        environment (class): all environment variables.
        parameters (_type_): all parameters of the model.

    Returns:
        purchases_to_predict (pd.DataFrame): All the purchases to predict.
        old_purchases (pd.DataFrame): All the purchases to insert in the graph.
        customers (pd.DataFrame): Customers features.
        articles (pd.DataFrame): Articles features.
    """

    # Load transactions dataset.
    transactions = pd.read_pickle(environment.transactions_path)

    old_purchases = transactions[(transactions['week_number'] > 0) & (
        transactions['week_number'] <= parameters['weeks_of_purchases'])]
    purchases_to_predict = transactions[transactions['week_number'] == 0]

    del transactions

    # Load customer and article features.
    customers = pd.read_pickle('pickles/customers_second_iteration.pkl')
    articles = pd.read_pickle('pickles/articles_second_iteration.pkl')

    # Split customers into train / valid / test set through a field `set` with
    # values 0 / 1 / 2.
    customer_id_train, customer_id_test = train_test_split(
        purchases_to_predict.customer_id.unique(), test_size=parameters['test_size'])
    customer_id_train, customer_id_valid = train_test_split(
        customer_id_train, test_size=parameters['valid_size'])

    test_customers = pd.DataFrame(customer_id_valid, columns=['customer_id'])
    test_customers['set'] = 2
    valid_customers = pd.DataFrame(customer_id_valid, columns=['customer_id'])
    valid_customers['set'] = 1

    customers = customers.merge(test_customers, on='customer_id', how='left')
    customers = customers.merge(valid_customers, on='customer_id', how='left')
    customers['set'].fillna(0, inplace=True)

    # Only process  customers who have transactions.
    customer_id = pd.Series(
        old_purchases.customer_id.unique() +
        purchases_to_predict.customer_id.unique()).rename('customer_id')
    customers = customers.merge(customer_id, on='customer_id', how='right')

    # Only process concerned articles who have transactions.
    article_id = pd.Series(
        old_purchases.article_id.unique() +
        purchases_to_predict.article_id.unique()).rename('article_id')
    articles = articles.merge(article_id, on='article_id', how='right')

    # Change indexes types in order to save memory.
    articles = articles.reset_index().rename({"index": "article_nid"}, axis=1)
    customers = customers.reset_index().rename(
        {"index": "customer_nid"}, axis=1)

    articles['article_nid'] = articles['article_nid'].astype('int32')
    customers['customer_nid'] = customers['customer_nid'].astype('int32')

    # Update transaction lists with new IDs and validation set Tag.
    old_purchases = old_purchases.merge(
        articles[['article_id', 'article_nid']], on='article_id', how='left')
    old_purchases = old_purchases.merge(
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

    purchases_to_predict['eid'] = purchases_to_predict['eid'].astype('int32')

    # Group the old purchases in order to have one edge per (article,
    # customer), with some extra columns.
    old_purchases = old_purchases.groupby(
        ['customer_nid', 'article_nid'], as_index=False
    ).agg(
        sales_channel_id=('sales_channel_id', 'mean'),
        first=('day_number', 'min'),
        last=('day_number', 'max'),
        purchases=('article_nid', 'count'),
    )

    old_purchases['mean_interval'] = (
        (old_purchases['last'] - old_purchases['first'])
        / old_purchases['purchases']
    ).astype("int32")

    return purchases_to_predict, old_purchases, customers, articles
