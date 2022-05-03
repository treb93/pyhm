import torch


class Environment:
    def __init__(self):
        self.transactions_path = "../pickles/transactions.pkl"
        self.customers_path = "../pickles/customers_gnn_full.pkl"
        self.articles_path = "../pickles/articles_gnn_medium.pkl"

        self.result_filepath = 'outputs/result_train.txt'

        #self.model_path = 'models/FULL_Precision_180_Epochs_0.03_2022-05-0219:25.pth'
        self.model_path = 0

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')
        #self.device = torch.device('cpu')

