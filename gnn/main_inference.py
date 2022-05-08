import math
from re import sub

import click
import dgl
import pandas as pd
import numpy as np
import torch
from environment import Environment
from gnn.src.metrics import get_recommendation_nids
from src.classes.dataloaders import DataLoaders
from src.classes.dataset import Dataset
from src.classes.graphs import Graphs
from src.get_dimension_dictionnary import get_dimension_dictionnary
from parameters import Parameters

from src.model.conv_model import ConvModel


def inference_ondemand(user_ids,  # List or 'all'
                       environment: Environment,
                       parameters: Parameters,
                       ):
    """
    Given a fully trained model, return recommendations specific to each user.

    Files needed to run
    -------------------
    Params used when training the model:
        Those params will indicate how to run inference on the model. Usually, they are outputted during training
        (and hyperparametrization).
    If using a saved already bought dict:
        The already bought dict: the dict includes all previous purchases of all user ids for which recommendations
                                 were requested. If not using a saved dict, it will be created using the graph.
                                 Using a saved already bought dict is not necessary, but might make the inference
                                 process faster.
    A) If using a saved graph:
        The saved graph: the graph that must include all user ids for which recommendations were requested. Usually,
                         it is outputted during training. It could also be created by another independent function.
        ID mapping: ctm_id and pdt_id mapping that allows to associate real-world information, e.g. item and customer
        identifier, to actual nodes in the graph. They are usually saved when generating a graph.
    B) If not using a saved graph:
        The graph will be generated on demand, using all the files in DataPaths of src.utils_data. All those files will
        be needed.

    Parameters
    ----------
    See click options below for details.

    Returns
    -------
    Recommendations for all user ids.

    """

    # TODO: Implement on demand.
    
    print("Load dataset.")
    
    # Create full train set
    dataset = Dataset(
        environment, parameters
    )

    print("Initialize graphs.")
    # Initialize graph & features
    graphs = Graphs(dataset, parameters)
    
    

    dim_dict = get_dimension_dictionnary(graphs, parameters)



    print("Build model.")
    # Initialize model
    model = ConvModel(dim_dict, parameters)

    print("Import model.")
    model.load_state_dict(
        torch.load(environment.model_path)
    )

    print("Initialize Dataloaders.")
    dataloaders = DataLoaders(graphs, dataset, parameters, environment)
    
    

    model.eval()
    with torch.no_grad():
        
        # Get embeddings for all graph.
        for input_nodes, output_nodes, blocks in dataloaders.dataloader_embedding:
        
            embeddings = model.get_embeddings(blocks, blocks[0].srcdata['features'])
            
            graphs.prediction_graph.nodes['article'].data['h'][output_nodes['article']] = embeddings['article']
            graphs.prediction_graph.nodes['customer'].data['h'][output_nodes['customer']] = embeddings['customer']
            
        
        # Process recommandations
        
        customers_per_batch = 200
        current_index = 0
        length = graphs.history_graph.num_nodes('customer')

        recommendation_chunks = []
        recommandation_dataframes = []
        
        # Backup nid indexes to disk.
        customer_index = dataset.customers[['customer_id', 'customer_nid']]
        customer_index.to_pickle('pickles/gnn_customer_index.pkl')
        
        article_index = dataset.articles[['article_id', 'article_nid']]
        article_index.to_pickle('pickles/gnn_article_index.pkl')

        while current_index < length :
            
            # TODO: les chopper autrement ?
            customer_nids = dataset.customers.loc[current_index: current_index + customers_per_batch, 'customer_nid'].values
            
            print(f"\rProcessing train recommendations for customers {current_index} - {current_index + customers_per_batch}            ", end = "")
            new_recommendations = get_recommendation_nids({
                'article': graphs.prediction_graph.nodes['article'].data['h'].to(environment.device),
                'customer': graphs.prediction_graph.nodes['customer'].data['h'][customer_nids].to(environment.device),
            }, parameters, environment, cutoff = max(parameters.precision_cutoffs), model = model)
            
            
            recommendation_chunks.append(
                torch.cat([customer_nids, new_recommendations], dim = 1).numpy()
            )

            # Backup chunks every 100 000 users.
            if current_index % 100000 == 99999 or current_index + customers_per_batch > len(dataset.customers):
                print("Backup recommandations for customers {}")
                dataframe = pd.DataFrame(np.concatenate(recommendation_chunks))
                
                recommendation_chunks = []
                
                recommandation_dataframes.append(dataframe)
                dataframe.to_pickle(f"pickles/gnn_recommandations_{len(recommandation_dataframes)}.pkl")
            
            current_index += customers_per_batch

    # Prepare dataframes.
    print("Concatenate and prepare dataframes.")
    recommandations = pd.concat(recommandation_dataframes)
    
    # Add real customers ID.
    recommandations = recommandations.merge(customer_index, left_on = "0", right_on = "customer_nid", on = "left")
    
    # Add real articles IDs
    for i in range(12):
        recommandations = recommandations.merge(article_index, left_on = f"{i+1}", right_on = "article_nid", on = "left")
        recommandations.rename({f"article_id": "article_{i}"}, axis = 1, inplace = True)
        
    # Remove unused columns
    recommandations.drop(columns = [i for i in range(13)], axis = 1, inplace = True)
    
    # Save expanded recommandation list.
    recommandations.to_pickle("pickles/gnn_recommandations_expanded.pkl")
    
    # Save compiled submission.

    recommandations['prediction'] = recommandations.apply(lambda x: [x[f"article_{i}"] for i in range (12)], axis = 1)
    
    submission = pd.read_pickle(environment.customers_path)
    submission = submission[['customer_id']]
    submission = submission.merge(
        recommandations[['customer_id', 'prediction']], how='left', on='customer_id')

    submission.to_csv('../submission_gnn.csv', index=False)
    
    return


@click.command()
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('--user_ids',
              multiple=True,
              default=['all'],
              help="IDs of users for which to generate recommendations. Either list of user ids, or 'all'.")
@click.option('--use_saved_graph', count=True,
              help='If true, will use graph that was saved on disk. Need to import ID mapping for users & items.')
@click.option('--trained_model_path', default='model.pth',
              help='Path where fully trained model is saved.')
@click.option('--use_saved_already_bought', count=True,
              help='If true, will use already bought dict that was saved on disk.')
@click.option('--graph_path', default='graph.bin',
              help='Path where the graph was saved. Mandatory if use_saved_graph is True.')
@click.option('--ctm_id_path', default='ctm_id.pkl',
              help='Path where the mapping for customer was save. Mandatory if use_saved_graph is True.')
@click.option('--pdt_id_path', default='pdt_id.pkl',
              help='Path where the mapping for items was save. Mandatory if use_saved_graph is True.')
@click.option('--already_bought_path', default='already_bought.pkl',
              help='Path where the already bought dict was saved. Mandatory if use_saved_already_bought is True.')
@click.option('--k', default=12,
              help='Number of recs to generate for each user.')
@click.option('--remove', default=0.99,
              help='Percentage of users to remove from graph if used_saved_graph = True. If more than 0, user_ids might'
                   ' not be in the graph. However, higher "remove" allows for faster inference.')
def main(params_path, user_ids, use_saved_graph, trained_model_path,
         use_saved_already_bought, graph_path, ctm_id_path, pdt_id_path,
         already_bought_path, k, remove):
    
    
    environment = Environment()
    parameters = Parameters({
        'embedding_on_full_set': True
    })

    inference_ondemand(user_ids,  # List or 'all'
                       environment,
                       parameters
    )


if __name__ == '__main__':
    main()
