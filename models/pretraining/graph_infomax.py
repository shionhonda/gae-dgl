from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch.nn import LayerNorm
from torch_geometric.nn.models import DeepGraphInfomax
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def custom_corruption(training_set: DataLoader) -> Data:
    train_sample = RandomSampler(
        training_set,
        replacement=False,
        num_samples=1,
        generator=None
    )
    return train_sample


class DeepGraphInfomax(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 encoder: torch.nn.Module,
                 normalize_hidden: bool = True,
                 summary: Callable = None,
                 corruption: Callable = None,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)

        self.dgi = DeepGraphInfomax(
            hidden_channels,
            encoder,
            summary,
            corruption
        )

        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.dropout = dropout

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi(x, edge_index)

        if self.norm is not None:
            summary = self.norm(summary).relu()
        else:
            summary = F.relu(summary)

        # Apply dropout
        summary = F.dropout(summary, p=self.dropout, training=self.training)
        # Apply second projection if required
        if self.lin2 is not None:
            summary = self.lin2(summary)

        return summary
