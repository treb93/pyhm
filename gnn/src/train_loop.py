from datetime import timedelta

import datetime
import time

import dgl
import numpy as np
import torch
from environment import Environment
from src.get_overall_metrics import get_overall_metrics
from src.classes.graphs import Graphs
from parameters import Parameters
from src.classes.dataloaders import DataLoaders
from src.classes.dataset import Dataset
from src.model.conv_model import ConvModel

from src.metrics import get_recommendation_nids, precision_at_k
from src.utils import save_txt

import gc


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
    model.train_precision_lists = [[] for i in parameters.precision_cutoffs] # We monitor cutoffs 6, 12, 24. 
    model.val_loss_list = []
    model.val_precision_lists = [[] for i in parameters.precision_cutoffs] # We monitor cutoffs 6, 12, 24. 
    best_metrics = {}  # For visualization
    max_metric = -0.1
    patience_counter = 0  # For early stopping
    min_loss = 1.1

    opt = parameters.optimizer(model.parameters(),
                               lr=parameters.lr)

    # Assign prediction layer to GPU.
    model.prediction_fn.to(environment.device)


    # if parameters.use_neighbor_sampling:
    #     model.to(environment.device)

    # TRAINING
    print('Starting training.')
    
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
            
            # Process embeddings and save it to graph.
            # if parameters.neighbor_sampling:
            #     # for b in range(len(blocks)):
            #     #     blocks[b] = blocks[b].to(environment.device)
            #     
# 
            #     embeddings = model.get_embeddings(blocks, blocks[0].srcdata['features'])
            #     
            #     # Update graph with new embeddings.
            #     graphs.full_graph.nodes['article'].data['h'][article_nids] = embeddings['article']
            #     graphs.full_graph.nodes['customer'].data['h'][customer_nids] = embeddings['customer']
            #     
            #     # Update batch graphs.
            #     pos_g.nodes['article'].data['h'] = embeddings['article']
            #     pos_g.nodes['customer'].data['h'] = embeddings['customer']
            #     
            #     neg_g.nodes['article'].data['h'] = embeddings['article']
            #     neg_g.nodes['customer'].data['h'] = embeddings['customer']
            #     
            # else:
            # Process embeddings at n iterations only.
            if (i % parameters.batches_per_embedding == 1) or (i == dataloaders.num_batches_train):
                print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Get embeddings...                   ", end ="")
                
                if i > 1:
                    del embeddings

                embeddings = model.get_embeddings(graphs.history_graph, {
                    'article': graphs.history_graph.nodes['article'].data['features'],
                    'customer': graphs.history_graph.nodes['customer'].data['features'],
                })
            
 
            

                
            print(f"\rTrain batch {i} / {dataloaders.num_batches_train} : Get scores...                   ", end ="")

            pos_score, neg_score = model(pos_g, neg_g, embeddings)
            pos_score_cat = torch.cat([pos_score_cat, pos_score])
            neg_score_cat = torch.cat([neg_score_cat, neg_score])
           
            
            # Proceed to gradient descent. 
            if (i % parameters.batches_per_embedding == 0) or (i == dataloaders.num_batches_train):
                #or parameters.neighbor_sampling 
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

        del embeddings

        print("\r Process valid batches...              ", end="")
        if get_metrics:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                i = 0
                
                # Refresh embeddins without dropout.
                # if parameters.neighbor_sampling == False:
                embeddings = model.get_embeddings(graphs.history_graph, {
                        'article': graphs.history_graph.nodes['article'].data['features'],
                        'customer': graphs.history_graph.nodes['customer'].data['features'],
                    })
 

                for _, pos_g, neg_g, blocks in dataloaders.dataloader_valid_loss:
                    i += 1
                    print(f"\rProcess valid batch {i} / {dataloaders.num_batches_valid}             ", end ="")

                    # if parameters.neighbor_sampling:
                    #     # Process embeddings from batch.
                    #     embeddings = model.get_embeddings(blocks, blocks[0].srcdata['features'])
                    #     
                    #     # Update positive graph.
                    #     pos_g.nodes['article'].data['h'] = embeddings['article']
                    #     pos_g.nodes['customer'].data['h'] = embeddings['customer']
# 
                    pos_g.nodes['article'].data['h'] = embeddings['article'][pos_g.nodes['article'].data['_ID'].long()]
                    pos_g.nodes['customer'].data['h'] = embeddings['customer'][pos_g.nodes['customer'].data['_ID'].long()]
                     
                    pos_score = model.prediction_fn(pos_g)
                    
                    neg_g.nodes['article'].data['h'] = embeddings['article'][neg_g.nodes['article'].data['_ID'].long()]
                    neg_g.nodes['customer'].data['h'] = embeddings['customer'][neg_g.nodes['customer'].data['_ID'].long()]
                    
                    neg_score = model.prediction_fn(neg_g)

                    val_loss = loss_fn(pos_score,
                                    neg_score,
                                    parameters=parameters,
                                    environment=environment
                                    )
                    total_loss += val_loss.item()

                valid_avg_loss = total_loss / i
                model.val_loss_list.append(valid_avg_loss)

        # Update graph.
        graphs.prediction_graph.nodes['article'].data['h'] = embeddings['article'][0:graphs.prediction_graph.num_nodes('article')]
        graphs.prediction_graph.nodes['customer'].data['h'] = embeddings['customer'][0:graphs.prediction_graph.num_nodes('customer')]
        del embeddings

        ############
        # METRICS
        if get_metrics and (epoch % 10 == 9 or epoch == parameters.num_epochs - 1):
                
            model.eval()
            with torch.no_grad():
                
                train_precision_at_k = get_overall_metrics(dataset.customers_nid_train, dataset, graphs, model, parameters, environment)
                valid_precision_at_k = get_overall_metrics(dataset.customers_nid_valid, dataset, graphs, model, parameters, environment)
                
                sentence = f"\rEpoch {parameters.start_epoch + epoch:05d} || TRAINING Loss {train_avg_loss:.5f} | Precision at 3 / 6 / 12 - {train_precision_at_k[0] * 100:.3f}% / {train_precision_at_k[1] * 100:.3f}% / {train_precision_at_k[2] * 100:.3f}% || VALIDATION Loss {valid_avg_loss:.5f} | Precision at 3 / 6 / 12 - {valid_precision_at_k[0] * 100:.3f}% / {valid_precision_at_k[1] * 100:.3f}% / {valid_precision_at_k[2] * 100:.3f}% "
                print(sentence)
                save_txt(sentence, environment.result_filepath, mode='a')
        
                for i in range(len(train_precision_at_k)) :
                    model.train_precision_lists[i].append(train_precision_at_k[i] * 100)
                    model.val_precision_lists[i].append(valid_precision_at_k[i] * 100)

                # Visualization of best metric
                if valid_precision_at_k[1] > max_metric:
                    max_metric = valid_precision_at_k[1]
                    best_metrics = {
                        'precision_3': valid_precision_at_k[0],
                        'precision_6': valid_precision_at_k[1],
                        'precision_12': valid_precision_at_k[2]
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
            best_metrics['lost_patience'] = True
            break

        elapsed = time.time() - start_time
        result_to_save = f'Epoch took {timedelta(seconds=elapsed)} \n'
        print(result_to_save)
        save_txt(result_to_save, environment.result_filepath, mode='a')
        
        # Clear memory.
        gc.collect()
        torch.cuda.empty_cache()

    viz = {'train_loss_list': model.train_loss_list,
           'train_precision_lists': model.train_precision_lists,
           'val_loss_list': model.val_loss_list,
           'val_precision_lists': model.val_precision_lists}

    best_metrics['min_loss'] = min_loss

    print('Training completed.')
    return model, viz, best_metrics
