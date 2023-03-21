
"""
title: Train a Graph Attention Network v2 (GATv2) 
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.graphs.gat.experiment import Configs as GATConfigs
#from labml_nn.graphs.gatv2 import GraphAttentionV2Layer
from gatModule import GATLayer



class GAT(Module):
    def __init__(self,
                 in_features: int,
                 n_hidden: int,
                 n_classes: int,
                 n_heads: int,
                 dropout: float,
                 share_weights: bool=True):
        
        super().__init__()
        
        self.layer1 = GATLayer(in_features, n_hidden, n_heads, is_concat=False, dropout=dropout, share_weights=share_weights)
        
        self.activation = nn.ELU()
        self.output = GATLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout, share_weights=share_weights)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        
        return self.output(x, adj_mat)
    
class Configs(GATConfigs):
    share_weights: bool = False
    model: GAT = 'gat_model'
    
@option
def gat_model(c: Configs):
    return GAT(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout, c.share_weights).to(c.device)

def main():
    conf = Configs()
    experiment.create(name='gat')
    
    experiment.configs(conf,{
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 0.05,
        'optimizer.weight_decay': 0.005,
        
        'dropout': 0.6,
    })
    
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()

