from sklearn.base import TransformerMixin
import pandas as pd


class AddUniclassFeatures(TransformerMixin):
    def __init__(self, articles, customers):
        self.articles = articles
        self.customers = customers
        return
    
    def fit(self, dataset):
        return self
        
    def transform(self, dataset):
        dataset = dataset.merge(self.articles, on = 'article_id', how = 'left')
        dataset = dataset.merge(self.customers, on = 'customer_id', how = 'left', suffixes = ('_article', '_customer'))

        print("Ajout du score d'âge")
        data_train_uniclass['age_ratio'] = data_train_uniclass.swifter.apply(lambda x: 
            x['age_around_15_customer'] * x['age_around_15_article'] +
            x['age_around_25_customer'] * x['age_around_25_article'] +
            x['age_around_35_customer'] * x['age_around_35_article'] +
            x['age_around_45_customer'] * x['age_around_45_article'] +
            x['age_around_55_customer'] * x['age_around_55_article'] +
            x['age_around_65_customer'] * x['age_around_65_article']
        , axis = 1)
            
        print("Ajout du score de catégorie")
        data_train_uniclass['index_ratio'] = data_train_uniclass.swifter.apply(lambda x: 
            x[x['index_group_name'].lower().split('/')[0]]
        , axis = 1)
        
        return dataset
        