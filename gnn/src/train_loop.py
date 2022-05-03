from datetime import timedelta

import datetime
import time

import dgl
import numpy as np
import torch
from environment import Environment
from src.classes.graphs import Graphs
from parameters import Parameters
from src.classes.dataloaders import DataLoaders
from src.classes.dataset import Dataset
from src.model.conv_model import ConvModel

from src.metrics import get_recommendation_nids, precision_at_k
from src.utils import save_txt


def train_loop(model: ConvModel,
               dataset: Dataset,
               graphs: Graphs,
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
    model.train_precision_lists = [[],[],[],[]] # We monitor cutoffs 6, 12, 24, 48. 
    model.val_loss_list = []
    model.val_precision_lists = [[],[],[],[]] # We monitor cutoffs 6, 12, 24, 48. 
    best_metrics = {}  # For visualization
    max_metric = -0.1
    patience_counter = 0  # For early stopping
    min_loss = 1.1

    opt = parameters.optimizer(model.parameters(),
                               lr=parameters.lr)

    # Assign prediction layer to GPU.
    model.prediction_fn.to(environment.device)

    # TRAINING
    print('Starting training.')
    
    embeddings = model.get_embeddings(graphs.history_graph, {
                    'article': graphs.history_graph.nodes['article'].data['features'],
                    'customer': graphs.history_graph.nodes['customer'].data['features'],
                })
    
    for epoch in range(parameters.num_epochs):
        start_time = time.time()
        model.train()  # Because if not, after eval, dropout would be still be inactive
        i = 0
        total_loss = 0
        
        opt.zero_grad()

        pos_score_cat = torch.tensor([])
        neg_score_cat = torch.tensor([])
                
        for _, pos_g, neg_g, blocks in dataloaders.dataloader_train_loss:
            i += 1
            
            # Process embeddings and initialize score tensors.
            if (i % parameters.batches_per_embedding == parameters.batches_per_embedding - 1) or (i == dataloaders.num_batches_train):
                print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Get embeddings...                   ", end ="")
                
                if i > 1:
                    del embeddings
                
                embeddings = model.get_embeddings(graphs.history_graph, {
                    'article': graphs.history_graph.nodes['article'].data['features'],
                    'customer': graphs.history_graph.nodes['customer'].data['features'],
                })
                
                print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Save embeddings on graph...                   ", end ="")
                graphs.prediction_graph.nodes['article'].data['h'] = embeddings['article'][0:graphs.prediction_graph.num_nodes('article')]
                graphs.prediction_graph.nodes['customer'].data['h'] = embeddings['customer'][0:graphs.prediction_graph.num_nodes('customer')]
                    
            
            
            print(f"\rProcess train batch {i} / {dataloaders.num_batches_train}                   ", end ="")

            pos_score, neg_score = model(pos_g, neg_g, embeddings)
            pos_score_cat = torch.cat([pos_score_cat, pos_score])
            neg_score_cat = torch.cat([neg_score_cat, neg_score])
           
                    
                    
            # Proceed to gradient descent. 
            if (i % parameters.batches_per_embedding == parameters.batches_per_embedding - 1) or (i == dataloaders.num_batches_train):
                print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Calculate loss...                   ", end ="")

                loss = loss_fn(pos_score,
                                neg_score,
                                parameters=parameters,
                                environment=environment
                                )

                total_loss += loss.item()
                
                print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Perform gradient descent...                   ", end ="")
                loss.backward()
                opt.step()
                
                del pos_score_cat
                del neg_score_cat
                
                pos_score_cat = torch.tensor([])
                neg_score_cat = torch.tensor([])
            
        
        train_avg_loss = total_loss / i
        model.train_loss_list.append(train_avg_loss)

        print("\r Process valid batches...              ", end="")
        if get_metrics:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                i = 0

        
                for _, pos_g, neg_g, blocks in dataloaders.dataloader_valid_loss:
                    i += 1
                    print(f"\rProcess valid batch {i} / {dataloaders.num_batches_valid}             ", end ="")

                    pos_g
                    neg_g
                    
                    pos_score = model.prediction_fn(pos_g)
                    
                    neg_g.nodes['article'].data['h'] = graphs.prediction_graph.nodes['article'].data['h'][neg_g.nodes['article'].data['_ID'].long()]
                    neg_g.nodes['customer'].data['h'] = graphs.prediction_graph.nodes['customer'].data['h'][neg_g.nodes['customer'].data['_ID'].long()]
                    
                    neg_score = model.prediction_fn(neg_g)

                    val_loss = loss_fn(pos_score,
                                    neg_score,
                                    parameters=parameters,
                                    environment=environment
                                    )
                    total_loss += val_loss.item()

                valid_avg_loss = total_loss / i
                model.val_loss_list.append(valid_avg_loss)

        ############
        # METRICS
        if get_metrics and (epoch % 10 == 5 or epoch == parameters.num_epochs - 1):
            
            model.eval()
            with torch.no_grad():
                
                customers_per_batch = 200
                current_index = 0
                length = len(dataset.customers_nid_train)

                recommendation_chunks = []

                while current_index < length :
                    
                    customer_nids = dataset.customers_nid_train[current_index: current_index + customers_per_batch]
                    
                    print(f"\rProcessing train recommendations for customers {current_index} - {current_index + customers_per_batch}            ", end = "")
                    new_recommendations = get_recommendation_nids({
                        'article': graphs.prediction_graph.nodes['article'].data['h'].to(environment.device),
                        'customer': graphs.prediction_graph.nodes['customer'].data['h'][customer_nids].to(environment.device),
                    }, parameters, environment, cutoff = 48)
                    
                    recommendation_chunks.append(new_recommendations)

                    customer_nids = range(current_index, current_index + customers_per_batch)


                    if current_index % 5000 == 0 or current_index + customers_per_batch < length:
                        recommendations = torch.cat(recommendation_chunks, dim = 0)
                        
                        precision = precision_at_k(recommendations, customer_nids, dataset)
                        
                        if current_index == 0:
                            precision_list = np.array([precision])
                        else: 
                            precision_list = np.append(precision_list, [precision], axis = 0)
                        
                        recommendation_chunks = []
                    
                    current_index += customers_per_batch
                      
                train_precision_at_k = np.mean(precision_list, axis = 0)


                # validation metrics
                
                batch_index = 0
                
                customers_per_batch = 200
                current_index = 0
                length = len(dataset.customers_nid_valid)

                recommendation_chunks = []

                while current_index < length :
                    
                    customer_nids = dataset.customers_nid_valid[current_index: current_index + customers_per_batch]
                    
                    print(f"\rProcessing valid recommendations for customers {current_index} - {current_index + customers_per_batch}                     ", end = "")
                    new_recommendations = get_recommendation_nids({
                        'article': graphs.prediction_graph.nodes['article'].data['h'].to(environment.device),
                        'customer': graphs.prediction_graph.nodes['customer'].data['h'][customer_nids].to(environment.device),
                    }, parameters, environment, cutoff = 48)
                    
                    recommendation_chunks.append(new_recommendations)

                    if current_index % 5000 == 0:
                        recommendations = torch.cat(recommendation_chunks, dim = 0)
                        
                        precision = precision_at_k(recommendations, customer_nids, dataset)
                                                
                        if current_index == 0:
                            precision_list = np.array([precision])
                        else: 
                            precision_list = np.append(precision_list, [precision], axis = 0)
                        
                        recommendation_chunks = []
                    
                    current_index += customers_per_batch
                    
                valid_precision_at_k = np.mean(precision_list, axis = 0)
                
                sentence = f"\rEpoch {parameters.start_epoch + epoch:05d} || TRAINING Loss {train_avg_loss:.5f} | Precision at 6 / 12 / 24 / 48 - {train_precision_at_k[0] * 100:.3f}% / {train_precision_at_k[1] * 100:.3f}% / {train_precision_at_k[2] * 100:.3f}% / {train_precision_at_k[3] * 100:.3f}% || VALIDATION Loss {valid_avg_loss:.5f} | Precision at 6 / 12 / 24 / 48 - {valid_precision_at_k[0] * 100:.3f}% / {valid_precision_at_k[1] * 100:.3f}% / {valid_precision_at_k[2] * 100:.3f}% / {valid_precision_at_k[3] * 100:.3f}% "
                print(sentence)
                save_txt(sentence, environment.result_filepath, mode='a')
        
                for i in range(len(train_precision_at_k)) :
                    model.train_precision_lists[i].append(train_precision_at_k[i] * 100)
                    model.val_precision_lists[i].append(valid_precision_at_k[i] * 100)

                # Visualization of best metric
                if np.sum(valid_precision_at_k) / 4 > max_metric:
                    max_metric = np.sum(valid_precision_at_k) / 4
                    best_metrics = {
                        'precision_6': valid_precision_at_k[0],
                        'precision_12': valid_precision_at_k[1],
                        'precision_24': valid_precision_at_k[2],
                        'precision_48': valid_precision_at_k[3],
                    }

            print("Save model.")
            date = str(datetime.datetime.now())[:-10].replace(' ', '')
            torch.save(
                model.state_dict(),
                f'models/FULL_Precision_{epoch}_Epochs_{valid_precision_at_k[1] * 100:.2f}_{date}.pth')
        else:
            sentence = f"Epoch {epoch:05d} | Training Loss {train_avg_loss:.5f} | Validation Loss {valid_avg_loss:.5f} | "
            print(sentence)
            save_txt(sentence, environment.result_filepath, mode='a')

        if valid_avg_loss < min_loss:
            min_loss = valid_avg_loss
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
           'train_precision_lists': model.train_precision_lists,
           'val_loss_list': model.val_loss_list,
           'val_precision_lists': model.val_precision_lists}

    print('Training completed.')
    return model, viz, best_metrics
