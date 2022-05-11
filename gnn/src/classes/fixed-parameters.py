
import torch


class FixedParameters:
    def __init__(self, num_epochs, start_epoch, patience, edge_batch_size,
                 remove, item_id_type, duplicates):
        """
        All parameters that are fixed, i.e. not part of the hyperparametrization.

        Attributes
        ----------
        ctm_id_type :
            Identifier for the customers.
        weeks_of_purchases (Days_of_clicks) :
            Number of days of purchases (clicks) that should be kept in the dataset.
            Intuition is that interactions of 12+ months ago might not be relevant. Max is 710 days
            Those that do not have any remaining interactions will be fed recommendations from another
            model.
        Discern_clicks :
            Clicks and purchases will be considered as 2 different edge types
        Duplicates :
            Determines how to handle duplicates in the training set. 'count_occurrence' will drop all
            duplicates except last, and the number of interactions will be stored in the edge feature.
            If duplicates == 'count_occurrence', aggregator_type needs to handle edge feature. 'keep_last'
            will drop all duplicates except last. 'keep_all' will conserve all duplicates.
        Explore :
            Print examples of recommendations and of similar sports
        Include_sport :
            Sports will be included in the graph, with 6 more relation types. User-practices-sport,
            item-utilizedby-sport, sport-belongsto-sport (and all their reverse relation type)
        item_id_type :
            Identifier for the items. Can be SPECIFIC ITEM IDENTIFIER (e.g. item SKU) or GENERIC ITEM IDENTIFIER
            (e.g. item family ID)
        Lifespan_of_items :
            Number of days since most recent transactions for an item to be considered by the
            model. Max is 710 days. Won't make a difference is it is > Days_of_interaction.
        Num_choices :
            Number of examples of recommendations and similar sports to print
        Patience :
            Number of epochs to wait for Early stopping
        Pred :
            Function that takes as input embedding of user and item, and outputs ratings. Choices : 'cos' for cosine
            similarity, 'nn' for multilayer perceptron with sigmoid function at the end
        Start_epoch :
            Load model from a previous epoch
        Train_on_clicks :
            When parametrizing the GNN, edges of purchases are always included. If true, clicks will also
            be included
        """
        self.ctm_id_type = 'CUSTOMER IDENTIFIER'
        self.weeks_of_purchases = 365  # Max is 710
        self.days_of_clicks = 30  # Max is 710
        self.discern_clicks = False
        self.duplicates = duplicates  # 'keep_last', 'keep_all', 'count_occurrence'
        self.edge_batch_size = edge_batch_size
        self.etype = [('customer', 'buys', 'article')]
        if self.discern_clicks:
            self.etype.append(('customer', 'clicks', 'article'))
        self.explore = True
        self.include_sport = False
        self.item_id_type = item_id_type
        self.k = 10
        self.lifespan_of_items = 180
        self.node_batch_size = 128
        self.num_choices = 10
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam
        self.patience = patience
        self.pred = 'cos'
        self.remove = remove
        self.remove_false_negative = True
        self.remove_on_inference = .7
        self.remove_train_eids = False
        self.report_model_coverage = False
        self.reverse_etype = {('customer', 'buys', 'article'): (
            'article', 'bought-by', 'customer')}
        if self.discern_clicks:
            self.reverse_etype[('customer', 'clicks', 'article')] = (
                'article', 'clicked-by', 'customer')
        self.run_inference = 1
        self.spt_id_type = 'sport_id'
        self.start_epoch = start_epoch
        self.subtrain_size = 0.05
        self.train_on_clicks = False
        self.valid_size = 0.1
        # self.dropout = .5  # HP
        # self.norm = False  # HP
        # self.use_popularity = False  # HP
        # self.days_popularity = 0  # HP
        # self.weight_popularity = 0.  # HP
        # self.use_recency = False  # HP
        # self.aggregator_type = 'mean_nn_edge'  # HP
        # self.aggregator_hetero = 'sum'  # HP
        # self.purchases_sample = .5  # HP
        # self.clicks_sample = .4  # HP
        # self.embedding_layer = False  # HP
        # self.edge_update = True  # Removed implementation; not useful
        # self.automatic_precision = False  # Removed implementation; not
        # useful
