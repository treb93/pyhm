import math
import datetime

import click
import numpy as np
import torch
from dgl.data.utils import save_graphs
from environment import Environment
from src.get_dimension_dictionnary import get_dimension_dictionnary
from src.classes.graphs import Graphs
from parameters import Parameters
from src.classes.dataloaders import DataLoaders
from src.classes.dataset import Dataset
from src.get_embeddings import get_embeddings
from src.max_margin_loss import max_margin_loss
from src.model.conv_model import ConvModel
from src.train_loop import train_loop

from src.utils_data import assign_graph_features
from src.utils import read_data, save_txt, save_outputs
from src.utils_vizualization import plot_train_loss
from src.metrics import (create_already_bought, create_ground_truth,
                         get_metrics_at_k, get_recommendation_tensor)
from src.evaluation import explore_recs, explore_sports, check_coverage
from presplit import presplit_data

from logging_config import get_logger

log = get_logger(__name__)

def launch_training(
    environment,
    parameters,
    visualization,
    check_embedding
):
    """
    Given the best hyperparameter combination, function to train the model on all available data.

    Files needed to run
    -------------------
    All the files in the Environment:
        It includes all the transactions, as well as features for customers and articles.
    Parameters found in hyperparametrization:
        Those params will indicate how to train the model. Usually, they are found when running the hyperparametrization
        loop.

    Parameters
    ----------
    See click options below for details.


    Saves to files
    --------------
    trained_model with its hyperparameters:
        The trained model with all parameters are saved to the folder 'models'.
    graph and ID mapping:
        When doing inference, it might be useful to import an already built graph (and the mapping that allows to
        associate node ID with personal information such as CUSTOMER IDENTIFIER or ITEM IDENTIFIER). Thus, the graph and ID mapping are saved to
        folder 'models'.
    """

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

    #model.load_state_dict(
    #    torch.load(
    #        "models/FULL_Recall_3.29_2022-04-1823:30.pth",
    #        map_location=environment.device))

    print("Initialize Dataloaders.")
    dataloaders = DataLoaders(graphs, dataset, parameters)

    # Run model
    hyperparameters_text = f'{str(parameters)} \n'

    save_txt(
        f'\n \n START - Hyperparameters \n{hyperparameters_text}',
        environment.result_filepath,
        "a")

    trained_model, viz, best_metrics = train_loop(
        model=model,
        graphs=graphs,
        dataset=dataset,
        dataloaders=dataloaders,
        loss_fn=max_margin_loss,
        get_metrics=True,
        parameters=parameters,
        environment=environment,
    )

    # Get viz & metrics
    if visualization:
        plot_train_loss(hyperparameters_text, viz)

    # Report performance on validation set
    sentence = f"BEST VALIDATION Precision {best_metrics['precision'] * 100:.3f}% "

    log.info(sentence)
    save_txt(sentence, environment.result_filepath, mode='a')

    # Report performance on test set
    log.debug('Test metrics start ...')
    trained_model.eval()
    with torch.no_grad():
        embeddings, node_ids = get_embeddings(graph,
                                              trained_model,
                                              dataloaders.dataloader_embedding,
                                              dataset.test_set_length,
                                              environment,
                                              parameters
                                              )

        precision = get_metrics_at_k(
            model=trained_model,
            y=embeddings,
            node_ids=node_ids,
            dataset=dataset,
            parameters=parameters
        )

        sentence = f"TEST Precision {precision:.3f}%"
        log.info(sentence)
        save_txt(sentence, environment.result_filepath, mode='a')

    if check_embedding:
        trained_model.eval()
        with torch.no_grad():
            log.debug('ANALYSIS OF RECOMMENDATIONS')

            recs = get_recommendation_tensor(
                embeddings,
                node_ids,
                trained_model,
                parameters
            )

            # TODO: réutiliser explore_recs ?

    # Save model
    date = str(datetime.datetime.now())[:-10].replace(' ', '')
    torch.save(trained_model.state_dict(),
               f'models/FULL_Recall_{precision * 100:.2f}_{date}.pth')
    # Save all necessary params
    save_outputs(
        {
            f'{date}_params': parameters,
            f'{date}_parameters': vars(parameters),
        },
        'models/'
    )
    print("Saved model & parameters to disk.")

    # Save graphs
    save_graphs(f'models/{date}_graph.bin', [graph])

    print("Saved graphs to disk.")


@click.command()
@click.option('--parameters_path', default='parameters.pkl',
               help='Path where the fixed parameters used in the hyperparametrization were saved.')
@click.option('--params_path', default='params.pkl',
               help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('-viz', '--visualization', count=True, help='Visualize result')
@click.option('--check_embedding', count=True,
               help='Explore embedding result')
@click.option('--remove',
               default=.99,
               help='Percentage of users to remove from train set. Ideally,'
               ' remove would be 0. However, higher "remove" accelerates training.')
@click.option('--batch_size', default=2048,
               help='Number of edges in a train / validation batch')
def main(
        parameters_path,
        params_path,
        visualization,
        check_embedding,
        remove,
        batch_size):

    environment = Environment()

    parameters = Parameters({
        'aggregator_hetero': 'mean',
        'aggregator_type': 'mean',
        'clicks_sample': 0.3,
        'delta': 0.266,
        'dropout': 0.01,
        'hidden_dim': 256,
        'out_dim': 128,
        'embedding_layer': True,
        'lr': 0.00017985194246308484,
        'n_layers': 5,
        'neg_sample_size': 2000,
        'norm': True,
        'use_popularity': True,
        'weight_popularity': 0.5,
        'days_popularity': 7,
        'purchases_sample': 0.5,
        'prediction_layer': 'cos',
        'use_recency': True,
        'num_workers': 4 if environment.cuda else 0
    })

    launch_training(
        environment=environment,
        parameters=parameters,
        visualization=visualization,
        check_embedding=check_embedding,
    )


if __name__ == '__main__':
    main()
