import math
import datetime

import click
import numpy as np
import torch
from dgl.data.utils import save_graphs
from gnn.environment import Environment
from gnn.parameters import Parameters
from gnn.src.classes.dataloaders import DataLoaders
from gnn.src.classes.dataset import Dataset
from gnn.src.create_graph import create_graph
from gnn.src.max_margin_loss import max_margin_loss
from gnn.src.train_loop import train_loop

from src.utils_data import assign_graph_features
from src.utils import read_data, save_txt, save_outputs
from src.model import ConvModel
from src.utils_vizualization import plot_train_loss
from src.metrics import (create_already_bought, create_ground_truth,
                         get_metrics_at_k, get_recommandation_tensor)
from src.evaluation import explore_recs, explore_sports, check_coverage
from presplit import presplit_data

from logging_config import get_logger

log = get_logger(__name__)

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')


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

    # Create full train set
    dataset = Dataset(
        environment, parameters
    )

    # Initialize graph & features
    graph = create_graph(dataset)

    dim_dict = {'user': graph.nodes['user'].data['features'].shape[1],
                'item': graph.nodes['item'].data['features'].shape[1],
                'out': parameters['out_dim'],
                'hidden': parameters['hidden_dim']}

    all_sids = None

    # Initialize model
    model = ConvModel(graph,
                      parameters['n_layers'],
                      dim_dict,
                      parameters['norm'],
                      parameters['dropout'],
                      parameters['aggregator_type'],
                      parameters['pred'],
                      parameters['aggregator_hetero'],
                      parameters['embedding_layer'],
                      )
    if cuda:
        model = model.to(device)

    model.load_state_dict(
        torch.load(
            "models/FULL_Recall_3.29_2022-04-1823:30.pth",
            map_location=device))

    # Initialize dataloaders
    # get training and test ids
    # (
    #    train_graph,
    #    train_eids_dict,
    #    valid_eids_dict,
    #    subtrain_uids,
    #    valid_uids,
    #    test_uids,
    #    all_iids,
    #    ground_truth_subtrain,
    #    ground_truth_valid,
    #    all_eids_dict
    # ) = train_valid_split(
    #    graph,
    #    data.ground_truth_test,
    #    parameters.etype,
    #    parameters.subtrain_size,
    #    parameters.valid_size,
    #    parameters.reverse_etype,
    #    parameters.train_on_clicks,
    #    parameters.remove_train_eids,
    #    parameters['clicks_sample'],
    #    parameters['purchases_sample'],
    # )

    dataloaders = DataLoaders(graph,
                              dataset,
                              parameters
                              )

    # Run model
    hyperparameters_text = f'{str(parameters)} \n'

    save_txt(
        f'\n \n START - Hyperparameters \n{hyperparameters_text}',
        environment.result_filepath,
        "a")

    trained_model, viz, best_metrics = train_loop(
        model,
        graph,
        dataset,
        dataloaders,
        max_margin_loss,
        get_metrics=True,
        parameters=parameters,
        environment=environment,
    )

    # Get viz & metrics
    if visualization:
        plot_train_loss(hyperparameters_text, viz)

    # Report performance on validation set
    sentence = ("BEST VALIDATION Precision "
                "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                .format(best_metrics['precision'] * 100,
                        best_metrics['recall'] * 100,
                        best_metrics['coverage'] * 100))

    log.info(sentence)
    save_txt(sentence, train_data_paths.result_filepath, mode='a')

    # Report performance on test set
    log.debug('Test metrics start ...')
    trained_model.eval()
    with torch.no_grad():
        embeddings, node_ids = get_embeddings(graph,
                                              parameters['out_dim'],
                                              trained_model,
                                              nodeloader_test,
                                              num_batches_test,
                                              cuda,
                                              device,
                                              parameters['embedding_layer'],
                                              )

        for ground_truth in [
                data.ground_truth_purchase_test,
                data.ground_truth_test]:
            precision = get_metrics_at_k(
                embeddings,
                graph,
                trained_model,
                parameters['out_dim'],
                ground_truth,
                all_eids_dict[('user', 'buys', 'item')],
                parameters.k,
                True,  # Remove already bought
                cuda,
                device,
                parameters['pred'],
                parameters['use_popularity'],
                parameters['weight_popularity'],
            )

            sentence = ("TEST Precision "
                        "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                        .format(precision * 100,
                                recall * 100,
                                coverage * 100))
            log.info(sentence)
            save_txt(sentence, train_data_paths.result_filepath, mode='a')

    if check_embedding:
        trained_model.eval()
        with torch.no_grad():
            log.debug('ANALYSIS OF RECOMMENDATIONS')
            if 'sport' in train_graph.ntypes:
                result_sport = explore_sports(embeddings,
                                              data.sport_feat_df,
                                              data.spt_id,
                                              parameters.num_choices)

                save_txt(
                    result_sport,
                    train_data_paths.result_filepath,
                    mode='a')

            already_bought_dict = create_already_bought(
                graph, all_eids_dict[('user', 'buys', 'item')], )
            already_clicked_dict = None
            if parameters.discern_clicks:
                already_clicked_dict = create_already_bought(
                    graph, all_eids_dict[('user', 'clicks', 'item')], etype='clicks', )

            users, items = data.ground_truth_test
            ground_truth_dict = create_ground_truth(users, items)
            user_ids = np.unique(users).tolist()

            recs = get_recommandation_tensor(
                embeddings,
                node_ids,
                trained_model,
                parameters
            )

            users, items = data.ground_truth_purchase_test
            ground_truth_purchase_dict = create_ground_truth(users, items)
            explore_recs(recs,
                         already_bought_dict,
                         already_clicked_dict,
                         ground_truth_dict,
                         ground_truth_purchase_dict,
                         data.item_feat_df,
                         parameters.num_choices,
                         data.pdt_id,
                         parameters.item_id_type,
                         train_data_paths.result_filepath)

            if parameters.item_id_type == 'SPECIFIC ITEM IDENTIFIER':
                coverage_metrics = check_coverage(data.user_item_train,
                                                  data.item_feat_df,
                                                  data.pdt_id,
                                                  recs)

                sentence = (
                    "COVERAGE \n|| All transactions : "
                    "Generic {:.1f}% | Junior {:.1f}% | Male {:.1f}% | Female {:.1f}% | Eco {:.1f}% "
                    "\n|| Recommendations : "
                    "Generic {:.1f}% | Junior {:.1f}% | Male {:.1f}% | Female {:.1f} | Eco {:.1f}%%" .format(
                        coverage_metrics['generic_mean_whole'] * 100,
                        coverage_metrics['junior_mean_whole'] * 100,
                        coverage_metrics['male_mean_whole'] * 100,
                        coverage_metrics['female_mean_whole'] * 100,
                        coverage_metrics['eco_mean_whole'] * 100,
                        coverage_metrics['generic_mean_recs'] * 100,
                        coverage_metrics['junior_mean_recs'] * 100,
                        coverage_metrics['male_mean_recs'] * 100,
                        coverage_metrics['female_mean_recs'] * 100,
                        coverage_metrics['eco_mean_recs'] * 100,
                    ))
                log.info(sentence)
                save_txt(sentence, train_data_paths.result_filepath, mode='a')

        save_outputs(
            {
                'embeddings': embeddings,
                'already_bought': already_bought_dict,
                'already_clicked': already_bought_dict,
                'ground_truth': ground_truth_dict,
                'recs': recs,
            },
            'outputs/'
        )

    # Save model
    date = str(datetime.datetime.now())[:-10].replace(' ', '')
    torch.save(trained_model.state_dict(),
               f'models/FULL_Recall_{recall * 100:.2f}_{date}.pth')
    # Save all necessary params
    save_outputs(
        {
            f'{date}_params': parameters,
            f'{date}_parameters': vars(parameters),
        },
        'models/'
    )
    print("Saved model & parameters to disk.")

    # Save graph & ID mapping
    save_graphs(f'models/{date}_graph.bin', [graph])
    save_outputs(
        {
            f'{date}_ctm_id': data.ctm_id,
            f'{date}_pdt_id': data.pdt_id,
        },
        'models/'
    )
    print("Saved graph & ID mapping to disk.")


@ click.command()
@ click.option('--parameters_path', default='parameters.pkl',
               help='Path where the fixed parameters used in the hyperparametrization were saved.')
@ click.option('--params_path', default='params.pkl',
               help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@ click.option('-viz', '--visualization', count=True, help='Visualize result')
@ click.option('--check_embedding', count=True,
               help='Explore embedding result')
@ click.option('--remove',
               default=.99,
               help='Percentage of users to remove from train set. Ideally,'
               ' remove would be 0. However, higher "remove" accelerates training.')
@ click.option('--batch_size', default=2048,
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
        'num_workers': 4 if cuda else 0
    })

    parameters.pop('remove', None)
    parameters.pop('batch_size', None)

    launch_training(
        environment=environment,
        parameters=parameters,
        visualization=visualization,
        check_embedding=check_embedding,
    )


if __name__ == '__main__':
    main()
