from typing import Optional
import torch
from torch_geometric.nn import GAE


class GAEV2(GAE):

    def __init__(self, encoder: torch.nn.Module, decoder: Optional[torch.nn.Module] = None):
        super(GAEV2, self).__init__(encoder=encoder, decoder=decoder)

    def forward(self, x, adj, sigmoid: bool = True):
        z = self.encode(x, adj)
        adj_rec = self.decode(z, sigmoid=sigmoid)
        return adj_rec

# def train_gae(gae: GAEV2, )



