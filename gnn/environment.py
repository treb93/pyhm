import torch


class Environment:
    def __init__(self):
        self.transactions_path = "../pickles/transactions.pkl"
        self.customers_path = "../pickles/customers_gnn_full.pkl"
        self.articles_path = "../pickles/articles_gnn_full.pkl"

        self.result_filepath = 'outputs/result_train.txt'

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')
