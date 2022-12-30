from typing import Union, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear
from torch_sparse import SparseTensor
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractproperty
import torch_geometric.transforms as T
# from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GroupAddRev, SAGEConv, GATv2Conv, GATConv, Aggregation
from torch_geometric.utils import index_to_mask


class SAGEConvBlock(torch.nn.Module):
    r"""A block containing a layer normalization and the GraphSAGE operator from the `"Inductive Representation
        Learning on Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

        .. math::
            \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
            \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

        If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
        projected via

        .. math::
            \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
            \mathbf{b})

        as described in Eq. (3) of the paper.

        Args:
            in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
                derive the size from the first input(s) to the forward method.
                A tuple corresponds to the sizes of source and target
                dimensionalities.
            out_channels (int): Size of each output sample.
            aggr (string or Aggregation, optional): The aggregation scheme to use.
                Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
                *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
                (default: :obj:`"mean"`)
            normalize (bool, optional): If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
            root_weight (bool, optional): If set to :obj:`False`, the layer will
                not add transformed root node features to the output.
                (default: :obj:`True`)
            project (bool, optional): If set to :obj:`True`, the layer will apply a
                linear transformation followed by an activation function before
                aggregation (as described in Eq. (3) of the paper).
                (default: :obj:`False`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.

        Shapes:
            - **inputs:**
              node features :math:`(|\mathcal{V}|, F_{in})` or
              :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
              if bipartite,
              edge indices :math:`(2, |\mathcal{E}|)`
            - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
              :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
        """

    def __init__(self, in_channels: Union[int, tuple[int]], out_channels: int, project: bool = False, bias: bool = True,
                 aggr: Optional[Union[str, list[str], Aggregation]] = "mean",  root_weight: bool = True, **kwargs):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.conv = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            normalize=False,
            root_weight=root_weight,
            project=project,
            bias=bias,
            **kwargs
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)


class GATConvBlock(torch.nn.Module):

    def __init__(self, in_channels: Union[int, tuple[int]], out_channels: int, version: str = "v2", heads: int = 1,
                 concat: bool = False, negative_slope: float = 0.2, dropout: float = 0.0, bias: bool = True,
                 add_self_loops: bool = True, edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean', **kwargs):
        r"""The graph attentional operator from the `"Graph Attention Networks"
            <https://arxiv.org/abs/1710.10903>`_ paper

            .. math::
                \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

            where the attention coefficients :math:`\alpha_{i,j}` are computed as

            .. math::
                \alpha_{i,j} =
                \frac{
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
                \right)\right)}
                {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
                \right)\right)}.

            If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
            the attention coefficients :math:`\alpha_{i,j}` are computed as

            .. math::
                \alpha_{i,j} =
                \frac{
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
                \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
                {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
                \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

            Args:
                in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
                    derive the size from the first input(s) to the forward method.
                    A tuple corresponds to the sizes of source and target
                    dimensionalities.
                out_channels (int): Size of each output sample.
                heads (int, optional): Number of multi-head-attentions.
                    (default: :obj:`1`)
                concat (bool, optional): If set to :obj:`False`, the multi-head
                    attentions are averaged instead of concatenated.
                    (default: :obj:`True`)
                negative_slope (float, optional): LeakyReLU angle of the negative
                    slope. (default: :obj:`0.2`)
                dropout (float, optional): Dropout probability of the normalized
                    attention coefficients which exposes each node to a stochastically
                    sampled neighborhood during training. (default: :obj:`0`)
                add_self_loops (bool, optional): If set to :obj:`False`, will not add
                    self-loops to the input graph. (default: :obj:`True`)
                edge_dim (int, optional): Edge feature dimensionality (in case
                    there are any). (default: :obj:`None`)
                fill_value (float or Tensor or str, optional): The way to generate
                    edge features of self-loops (in case :obj:`edge_dim != None`).
                    If given as :obj:`float` or :class:`torch.Tensor`, edge features of
                    self-loops will be directly given by :obj:`fill_value`.
                    If given as :obj:`str`, edge features of self-loops are computed by
                    aggregating all features of edges that point to the specific node,
                    according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
                    :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
                bias (bool, optional): If set to :obj:`False`, the layer will not learn
                    an additive bias. (default: :obj:`True`)
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.

            Shapes:
                - **input:**
                  node features :math:`(|\mathcal{V}|, F_{in})` or
                  :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
                  if bipartite,
                  edge indices :math:`(2, |\mathcal{E}|)`,
                  edge features :math:`(|\mathcal{E}|, D)` *(optional)*
                - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
                  :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
                  If :obj:`return_attention_weights=True`, then
                  :math:`((|\mathcal{V}|, H * F_{out}),
                  ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
                  or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
                  (|\mathcal{E}|, H)))` if bipartite
            """
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)

        if version == "v1":
            self.conv = GATConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
                fill_value=fill_value,
                bias=bias,
                **kwargs
            )
        elif version == "v2":
            self.conv = GATv2Conv(
                in_channels,
                out_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
                fill_value=fill_value,
                bias=bias,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown version '{version}'")

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)
