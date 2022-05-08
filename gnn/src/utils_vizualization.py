import matplotlib.pyplot as plt
from datetime import datetime
import textwrap

import numpy as np
from pytest import param

from parameters import Parameters


def plot_train_loss(hp_sentence, viz, parameters: Parameters):
    """
    Visualize train & validation loss & metrics. hp_sentence is used as the title of the plot.

    Saves plots in the plots folder.
    """
    if 'val_loss_list' in viz.keys():
        fig = plt.figure()
        x = np.arange(len(viz['train_loss_list']))
        plt.title('\n'.join(textwrap.wrap(hp_sentence, 60)))
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        plt.plot(x, viz['train_loss_list'])
        plt.plot(x, viz['val_loss_list'])
        plt.legend(['training loss', 'valid loss'], loc='upper left')
        plt.savefig('plots/' + str(datetime.now())[:-10] + 'loss.png')
        plt.close(fig)

    if 'train_precision_lists' in viz.keys():
        
        
        fig = plt.figure()
        
        print(viz['train_precision_lists'])
        
        x = np.arange(len(viz['train_precision_lists'][0]))
        plt.title('\n'.join(textwrap.wrap(hp_sentence, 60)))
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        
        for i in [0, 1]:   
            k = parameters.precision_cutoffs[i] 
            
            plt.plot(x, viz['train_precision_lists'][i], label = f"Train precision at {k}")
            plt.plot(x, viz['val_precision_lists'][i], label = f"Valid precision at {k}")
            
        plt.ylim(0, 8)
        plt.legend(loc='upper left')
        plt.savefig('plots/' + str(datetime.now())[:-10] + 'metrics.png')
        plt.close(fig)
