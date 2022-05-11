import math
from re import sub

import click
import dgl
import pandas as pd
import numpy as np
import torch
from environment import Environment
from src.get_overall_metrics import get_overall_metrics
from src.metrics import get_recommendation_nids, precision_at_k
from src.classes.dataloaders import DataLoaders
from src.classes.dataset import Dataset
from src.classes.graphs import Graphs
from src.get_dimension_dictionnary import get_dimension_dictionnary
from parameters import Parameters

from src.model.conv_model import ConvModel


def inference_ondemand(first_half,  
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
    
    #if first_half == True : 
    #    offset = 0
    #    customers = dataset.customers.iloc[0: 700000]
    #else :
    #    customers = dataset.customers.iloc[700000:len(dataset.customers)].copy()
    
    customers = dataset.customers
    
    model.eval()
    with torch.no_grad():
            
        customers_per_graph = 100000
        index_for_graphs = 0

        customer_embeddings = []
        recommandation_dataframes = []

        while index_for_graphs < len(customers):
            
            index_graph_end = min(len(customers), index_for_graphs + customers_per_graph)
            #index_graph_end = graphs.prediction_graph.num_nodes('customer')
            
            print(f"\rProcess subgraph for customers {index_for_graphs} - {index_graph_end}")
            print(f"Num articles in prediction: {graphs.prediction_graph.num_nodes('article')}")
                        
            subgraph = dgl.node_subgraph(graphs.history_graph, {
                'article': range(graphs.prediction_graph.num_nodes('article')),
                #'customer': customers.iloc[index_for_graphs:index_graph_end]['customer_nid'].values
                'customer': range(index_for_graphs, index_graph_end)
            })
            
            # Checks that the node IDS of the subgraph are in the same order and continuous.
            # assert (subgraph.nodes['customer'].data['_ID'].numpy() == range(index_for_graphs, index_graph_end)).min()
            # assert (subgraph.nodes['article'].data['_ID'].numpy() == range(0, len(dataset.articles['article_nid']))).min()
    
            embeddings = model.get_embeddings(subgraph, {
                'article': subgraph.nodes['article'].data['features'],
                'customer': subgraph.nodes['customer'].data['features'],
            })
            
            articles_embeddings = embeddings['article']

            graphs.history_graph.nodes['article'].data['h'][0:graphs.prediction_graph.num_nodes('article')] = embeddings['article']
            graphs.history_graph.nodes['customer'].data['h'][index_for_graphs : index_graph_end] = embeddings['customer']

            
            
            # Process recommandations
            customers_per_batch = 200
            index_for_recommandations = index_for_graphs
            recommendation_chunks = []
            
            precision_list = np.array([])
            
            while index_for_recommandations < index_graph_end:
            
                index_batch_end = min(index_graph_end, index_for_recommandations + customers_per_batch)
                
                customer_nids = np.arange(index_for_recommandations, index_batch_end)
                
                print(f"\rProcessing train recommendations for customers {index_for_recommandations} - {index_batch_end}            ", end = "")
                new_recommendations = get_recommendation_nids({
                    'article': articles_embeddings.to(environment.device),
                    'customer': graphs.history_graph.nodes['customer'].data['h'][index_for_recommandations: index_batch_end].to(environment.device),
                }, parameters, environment, cutoff = max(parameters.precision_cutoffs), model = model)
                
                # precision = precision_at_k(new_recommendations, customer_nids, dataset, parameters)
                # 
                # if precision_list.shape[0] == 0:
                #     precision_list = np.array([precision])
                # else: 
                #     precision_list = np.append(precision_list, [precision], axis = 0)
                
                recommendation_chunks.append(
                    np.concatenate([customer_nids.reshape((-1, 1)), new_recommendations], axis = 1)
                )
                
                # Go to next recommandation batch.
                index_for_recommandations += customers_per_batch

            # Build recommandation chunks.
            print(f"\rChunk recommandations for customers {index_for_graphs} - {index_graph_end}")
            dataframe = pd.DataFrame(np.concatenate(recommendation_chunks))
            
            recommandation_dataframes.append(dataframe)
            # Save chunks.
            dataframe.to_pickle(f"../pickles/gnn_recommandations_{len(recommandation_dataframes)}.pkl")
            
            # Go to next graph.
            index_for_graphs += customers_per_graph
            
            
            
        # print("Save embeddings on dataset.", len(customer_embeddings))
        # customers['embeddings'] = customer_embeddings
            
        
        #customer_index = customers[['customer_id', 'customer_nid', 'embeddings']]
        customer_index = customers[['customer_id', 'customer_nid']]
        customer_index.to_pickle(f'../pickles/gnn_customer_index.pkl')
        
        #article_index = dataset.articles[['article_id', 'article_nid', 'embeddings']]
        article_index = dataset.articles[['article_id', 'article_nid']]
        article_index.to_pickle('../pickles/gnn_article_index.pkl')
        
        
    # Prepare dataframes.
    print("Concatenate and prepare dataframes.")
    recommandations = pd.concat(recommandation_dataframes)
    
    
    dataset.purchased_list.to_pickle(f"../pickles/gnn_purchase_list.pkl")
    
    recommandations.to_pickle(f"../pickles/gnn_recommandations_raw.pkl")
    
    # Add real customers ID.
    recommandations = recommandations.merge(customer_index[['customer_id', 'customer_nid']], left_on = 0, right_on = "customer_nid", how = "left")

    # Add real articles IDs
    for i in range(12):
        recommandations = recommandations.merge(article_index[['article_id', 'article_nid']], left_on = i+1, right_on = "article_nid", how = "left")
        recommandations.rename({f"article_id": f"article_{i}"}, axis = 1, inplace = True)
        
    # Remove unused columns
    recommandations.drop(columns = ['article_nid_x', 'article_nid_y'], axis = 1, inplace = True)
    recommandations.drop(columns = [i for i in range(13)], axis = 1, inplace = True)

    # Save expanded recommandation list.
    recommandations.to_pickle(f"../pickles/gnn_recommandations_expanded.pkl")

    # Save compiled submission.
    recommandations['prediction'] = recommandations.apply(lambda x: list([x[f"article_{i}"] for i in range (12)]), axis = 1)
    recommandations['prediction'] = recommandations['prediction'].apply(lambda x: ' '.join(x))

    submission = pd.read_pickle(environment.customers_path)
    submission = submission[['customer_id']]
    submission = submission.merge(
        recommandations[['customer_id', 'prediction']], how='left', on='customer_id')


    submission.to_csv(f"../submissions/submission_gnn.csv", index=False)
    
    return


@click.command()
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('--user_ids',
              multiple=True,
              default=['all'],
              help="IDs of users for which to generate recommendations. Either list of user ids, or 'all'.")
@click.option('--first_half', count=True)
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
def main(params_path, user_ids, first_half, trained_model_path,
         use_saved_already_bought, graph_path, ctm_id_path, pdt_id_path,
         already_bought_path, k, remove):
    
    
    environment = Environment()
    parameters = Parameters({
       'embedding_on_full_set': True,
       'include_last_week_in_history': True
    })

    inference_ondemand(first_half,  # 'all' | 'first-half' | 'last-half'
                       environment,
                       parameters
    )


if __name__ == '__main__':
    main()
