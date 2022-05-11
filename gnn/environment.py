import torch


class Environment:
    def __init__(self):
        self.transactions_path = "../pickles/transactions_without_outliers.pkl"
        self.customers_path = "../pickles/customers_gnn_full.pkl"
        self.articles_path = "../pickles/articles_gnn_full.pkl"

        self.result_filepath = 'outputs/result_train.txt'

        self.search_result_path = "../pickles/search_result_table.pkl"

        self.model_path = 'models/Modèle 1/FULL_Precision_2.30_2022-05-0900:38.pth'
        #self.model_path = 0

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')
        #self.device = torch.device('cpu')

