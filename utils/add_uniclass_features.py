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
        
        # Categorical fields
        categories = ["product_type_name", "product_group_name", 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name', 'club_member_status', 'fashion_news_frequency']

        for category in categories:
            dataset[category] = dataset[category].astype('category')
            
        return dataset
        