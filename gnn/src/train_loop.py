from datetime import timedelta

import datetime
import time

import dgl
import torch
from gnn.environment import Environment
from gnn.parameters import Parameters
from gnn.src.classes.dataloaders import DataLoaders
from gnn.src.classes.dataset import Dataset
from gnn.src.get_embeddings import get_embeddings
from gnn.src.model import ConvModel

from src.metrics import get_metrics_at_k
from src.utils import save_txt


def train_loop(model: ConvModel,
               dataset: Dataset,
               graph: dgl.DGLHeteroGraph,
               dataloaders: DataLoaders,
               loss_fn,
               parameters: Parameters,
               environment: Environment,
               get_metrics=False,
               ):
    """
    Main function to train a GNN, using max margin loss on positive and negative examples.

    Process:
        - A full training epoch
            - Batch by batch. 1 batch is composed of multiple computational blocks, required to compute embeddings
              for all the nodes related to the edges in the batch.
            - Input the initial features. Compute the embeddings & the positive and negative scores
            - Also compute other considerations for the loss function: negative mask, recency scores
            - Loss is returned, then backward, then step.
            - Metrics are computed on the subtraining set (using nodeloader)
        - Validation set
            - Loss is computed (in model.eval() mode) for validation edge for early stopping purposes
            - Also, metrics are computed on the validation set (using nodeloader)
        - Logging & early stopping
            - Everything is logged, best metrics are saved.
            - Using the patience parameter, early stopping is applied when val_loss stops going down.
    """
    model.train_loss_list = []
    model.train_precision_list = []
    model.train_recall_list = []
    model.train_coverage_list = []
    model.val_loss_list = []
    model.val_precision_list = []
    model.val_recall_list = []
    model.val_coverage_list = []
    best_metrics = {}  # For visualization
    max_metric = -0.1
    patience_counter = 0  # For early stopping
    min_loss = 1.1

    opt = parameters.optimizer(model.parameters(),
                               lr=parameters.lr)

    # TRAINING
    print('Starting training.')
    for epoch in range(parameters.start_epoch, parameters.num_epochs):
        start_time = time.time()
        print('TRAINING LOSS')
        model.train()  # Because if not, after eval, dropout would be still be inactive
        i = 0
        total_loss = 0
        for _, pos_g, neg_g, blocks in dataloaders.dataloader_train_loss:
            opt.zero_grad()

            if environment.cuda:
                blocks = [b.to(environment.device) for b in blocks]
                pos_g = pos_g.to(environment.device)
                neg_g = neg_g.to(environment.device)

            i += 1
            if i % 10 == 0:
                print(f"Edge batch {i} out of {dataloaders.num_batches_train}")

            input_features = blocks[0].srcdata['features']

            _, pos_score, neg_score = model(blocks,
                                            input_features,
                                            pos_g,
                                            neg_g,
                                            parameters.embedding_layer,
                                            )
            loss = loss_fn(pos_score,
                           neg_score,
                           parameters=parameters,
                           environment=environment
                           )

            if epoch > 0:  # For the epoch 0, no training (just report loss)
                loss.backward()
                opt.step()
            total_loss += loss.item()

            if epoch == 0 and i > 10:
                break  # For the epoch 0, report loss on only subset

        train_avg_loss = total_loss / i
        model.train_loss_list.append(train_avg_loss)

        print('VALIDATION LOSS')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            i = 0

            for _, pos_g, neg_g, blocks in dataloaders.dataloader_valid_loss:
                i += 1
                if i % 10 == 0:
                    print(
                        f"Edge batch {i} out of {dataloaders.num_batches_valid}")

                if environment.cuda:
                    blocks = [b.to(environment.device) for b in blocks]
                    pos_g = pos_g.to(environment.device)
                    neg_g = neg_g.to(environment.device)

                input_features = blocks[0].srcdata['features']
                _, pos_score, neg_score = model(blocks,
                                                input_features,
                                                pos_g,
                                                neg_g,
                                                parameters.embedding_layer,
                                                )

                val_loss = loss_fn(pos_score,
                                   neg_score,
                                   parameters=parameters,
                                   environment=environment
                                   )
                total_loss += val_loss.item()
                print(val_loss.item())

            val_avg_loss = total_loss / i
            model.val_loss_list.append(val_avg_loss)

        ############
        # METRICS PER EPOCH
        if get_metrics and epoch % 20 == 1:
            model.eval()
            with torch.no_grad():
                # training metrics
                print('TRAINING METRICS')
                # TODO: Sortir les batches de la fonction sans quoi ça va
                # coincer.
                y, node_ids = get_embeddings(graph,
                                             model,
                                             dataloaders,
                                             parameters=parameters,
                                             environment=environment
                                             )

                train_precision_at_k = get_metrics_at_k(
                    model,
                    y,
                    node_ids,
                    dataset,
                    parameters
                )

                # validation metrics
                print('VALIDATION METRICS')
                y, node_ids = get_embeddings(graph,
                                             model,
                                             dataloaders.dataloader_valid_metrics,
                                             environment=environment,
                                             parameters=parameters
                                             )

                val_precision_at_k = get_metrics_at_k(
                    model,
                    y,
                    node_ids,
                    dataset,
                    parameters
                )
                sentence = f"""Epoch {epoch: 05d} || TRAINING Loss {train_avg_loss: .5f} | Precision at k {train_precision_at_k * 100: .3f}%
                || VALIDATION Loss {val_avg_loss: .5f} | Precision {val_precision_at_k * 100: .3f}% """
                print(sentence)
                save_txt(sentence, environment.result_filepath, mode='a')

                model.train_precision_list.append(train_precision_at_k * 100)
                model.val_precision_list.append(val_precision_at_k * 100)

                # Visualization of best metric
                if val_precision_at_k > max_metric:
                    max_metric = val_precision_at_k
                    best_metrics = {
                        'precision': val_precision_at_k,
                    }

            print("Save model.")
            date = str(datetime.datetime.now())[:-10].replace(' ', '')
            torch.save(
                model.state_dict(),
                f'models/FULL_Precision_{val_precision_at_k * 100:.2f}_{date}.pth')
        else:
            sentence = "Epoch {epoch:05d} | Training Loss {train_avg_loss:.5f} | Validation Loss {val_avg_loss:.5f} | "
            print(sentence)
            save_txt(sentence, environment.result_filepath, mode='a')

        if val_avg_loss < min_loss:
            min_loss = val_avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == parameters.patience:
            print("Lost patience.")
            break

        elapsed = time.time() - start_time
        result_to_save = f'Epoch took {timedelta(seconds=elapsed)} \n'
        print(result_to_save)
        save_txt(result_to_save, environment.result_filepath, mode='a')

    viz = {'train_loss_list': model.train_loss_list,
           'train_precision_list': model.train_precision_list,
           'val_loss_list': model.val_loss_list,
           'val_precision_list': model.val_precision_list}

    print('Training completed.')
    return model, viz, best_metrics
