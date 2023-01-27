from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import GroupAddRev, Aggregation
from models.layers import SAGEConvBlock, GATConvBlock


class RevSAGEConvEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_convs: int = 1,
                 dropout: float = 0.0, project: bool = False, root_weight: bool = True,
                 aggr: Optional[Union[str, list[str], Aggregation]] = "mean",
                 num_groups: int = 2, normalize_hidden: bool = True):
        super().__init__()

        self.dropout = dropout
        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(hidden_channels, out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_groups, given {hidden_channels} and {num_groups}"
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_convs):
            conv = SAGEConvBlock(
                in_channels=hidden_channels // num_groups,
                out_channels=hidden_channels // num_groups,
                project=project,
                bias=True,
                aggr=aggr,
                root_weight=root_weight
            )
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x


class RevGATConvEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_convs: int = 1,
                 dropout: float = 0.0, version: str = "v2",
                 edge_dim: Optional[int] = None, heads: int = 1,
                 num_groups: int = 2, normalize_hidden: bool = True):
        super().__init__()

        self.dropout = dropout
        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.lin1 = None
        self.lin2 = None
        self.norm = None

        if in_channels != hidden_channels:
            self.lin1 = Linear(in_channels, hidden_channels)

        if hidden_channels != out_channels:
            self.lin2 = Linear(hidden_channels, out_channels)

        if normalize_hidden:
            self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_groups, given {hidden_channels} and {num_groups}"
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_convs):
            conv = GATConvBlock(
                in_channels=hidden_channels // num_groups,
                out_channels=hidden_channels // num_groups,
                version=version,
                heads=heads,
                edge_dim=edge_dim,
                bias=True,
                add_self_loops=True,
                negative_slope=0.2,
                concat=True,
                fill_value='mean'
            )
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    def reset_parameters(self):
        if self.lin1 is not None:
            self.lin1.reset_parameters()

        if self.lin2 is not None:
            self.lin2.reset_parameters()

        if self.norm is not None:
            self.norm.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        # Apply first projection if required
        if self.lin1 is not None:
            x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # Apply conv layers
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        # Normalize if required
        if self.norm is not None:
            x = self.norm(x).relu()
        else:
            x = F.relu(x)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second projection if required
        if self.lin2 is not None:
            x = self.lin2(x)

        return x
