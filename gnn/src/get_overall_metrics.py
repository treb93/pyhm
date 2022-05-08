# Reproduce loop
import numpy as np
from environment import Environment
from parameters import Parameters
from src.classes.dataset import Dataset
from src.classes.graphs import Graphs
from src.metrics import get_recommendation_nids, precision_at_k
from src.model.conv_model import ConvModel


def get_overall_metrics(
    customer_nids, 
    dataset: Dataset, 
    graphs: Graphs, 
    model: ConvModel,
    parameters: Parameters, 
    environment: Environment, 
    customers_per_batch = 200
):
    
    # Get embeddings for all articles to score.
    # if parameters.neighbor_sampling:
    #     article_embeddings = graphs.full_graph.nodes['article'].data['h'][article_nids].to(environment.device)
    # else: 
    
    
    #!! Assuming that the prediction graph article NIDs are the first ones of the dataset.
    article_embeddings = graphs.prediction_graph.nodes['article'].data['h'].to(environment.device) 
            
    current_index = 0
    length = len(customer_nids)

    recommendation_chunks = []
    customer_nids_chunks = []
    precision_list = np.array([])

    while current_index < length :
        
        batch_customer_nids = customer_nids[current_index: current_index + customers_per_batch]

        # if parameters.neighbor_sampling:
        #     customer_embeddings = graphs.full_graph.nodes['customer'].data['h'][batch_customer_nids].to(environment.device)
        # else: 
        customer_embeddings = graphs.prediction_graph.nodes['customer'].data['h'][batch_customer_nids].to(environment.device)

        print(f"\rProcessing recommendations for customers {current_index} - {current_index + customers_per_batch}                     ", end = "")
        new_recommendations = get_recommendation_nids({
                        'article': article_embeddings,
                        'customer': customer_embeddings,
                    }, parameters, environment, cutoff = max(parameters.precision_cutoffs), model = model)
        
        recommendation_chunks.append(new_recommendations)
        customer_nids_chunks.append(batch_customer_nids)

        # Calculate precision when the number of chunks is considered as optimal.
        if current_index % 5000 == 0 or current_index + customers_per_batch > length:
            
            recommendations = np.concatenate(recommendation_chunks, axis = 0)
            batch_customer_nids = np.concatenate(customer_nids_chunks, axis = 0)
            
            precision = precision_at_k(recommendations, batch_customer_nids, dataset, parameters)
            
            if precision_list.shape[0] == 0:
                precision_list = np.array([precision])
            else: 
                precision_list = np.append(precision_list, [precision], axis = 0)
            
            recommendation_chunks = []
            customer_nids_chunks = []
        
        current_index += customers_per_batch
        

    mean_precision = np.mean(precision_list, axis = 0)
    
    del recommendation_chunks
    del precision_list    
    
    return mean_precision