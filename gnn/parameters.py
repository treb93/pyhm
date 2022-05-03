import torch


class Parameters():
    def __init__(self, new_parameters):
        self.weeks_of_purchases = 53  # Max is 104
        self.duplicates = 'keep_all'  # 'keep_last', 'keep_all', 'count_occurrence'
        self.edge_batch_size = 4000000
        self.embedding_batch_size = 70000
        self.explore = True
        self.k = 12
        self.lifespan_of_items = 180
        self.num_choices = 10
        self.start_epoch = 0
        self.num_epochs = 200
        self.optimizer = torch.optim.Adam
        self.patience = 20
        self.prediction_layer = 'cos'
        self.remove = 0
        # TODO: Dégager remove_false_negative ?
        self.remove_false_negative = True
        self.remove_on_inference = .7
        self.report_model_coverage = True
        self.run_inference = 1
        self.subtrain_size = 0.05
        self.valid_size = 0.2

        self.aggregator_hetero = 'mean'
        self.aggregator_type = 'mean'
        self.clicks_sample = 0.3
        self.delta = 0.266
        self.dropout = 0.01
        self.hidden_dim = 256
        self.out_dim = 128
        self.embedding_layer = True
        self.lr = 0.00017985194246308484
        self.n_layers = 4
        self.neg_sample_size = 300
        self.norm = True
        self.use_popularity = True
        self.weight_popularity = 0.5
        self.days_popularity = 7
        self.purchases_sample = 0.5
        self.prediction_layer = 'cos'
        self.use_recency = True
        self.num_workers = 0
        self.partial_sampling_num_neighbors = 5
        
        self.customers_without_prediction_ratio = 0
        self.embedding_on_full_set = False
        self.batches_per_embedding = 3

        self.update(new_parameters)

    def update(self, new_parameters):
        for key in new_parameters.keys():
            setattr(self, key, new_parameters[key])
