

import torch 
from torch import nn

from labml_helpers.module import Module


class GATLayer(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = False,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = True):
        
        super().__init__()
        
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        
        if is_concat:
            assert out_features % n_heads == 0
            self.n_heads = out_features // n_heads
        else:
            self.n_hidden = out_features
        
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
            
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout = nn.Dropout(dropout)
        
    """
    h is the input node embeddings of shape [n_nodes, in_features]
    adj_mat is the adjacency matrix of shape [n_nodes, n_nodes, n_heads]
    We use shape [n_nodes, n_nodes, 1] since the adjacency is the same for each head
    Adjacency matrix represent the edges (or connections) among nodes. adj_mat[i][j] is True if there is an edge from node i to node j
    """
    
    def forward(self, h:torch.Tensor, adj_mat: torch.Tensor):
        n_nodes = h.shape=[0]
        
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        
        """
        calculate attention score
        e_ij = a(W_l * h_i, W_r * h_j) = a(g_li, g_rj)
        """
        
        g_l_repeat = g_l_repeat(n_nodes, 1, 1)
        
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        
        g_sum = g_l_repeat + g_r_repeat_interleave
        
        """
        reshapre so that g_sum[i, j] is g_li + g_ri
        """
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        
        """
        e is of shape [n_nodes, n_nodes, n_heads, 1]
        """
        e = self.attn(self.activation(g_sum))

        """
        remove the last dimension of size 1
        """
        e = e.squeeze(-1)
        
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        
        """
        Calculate final output for each head
        h'^k_i = sum(a^k_ij * g_r_j,k)
        """
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)
        
        """
        Concatenate the heads
        """
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)