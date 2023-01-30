from typing import Union, Optional
import torch
from torch import Tensor
from torch.nn import LayerNorm
from torch_geometric.nn.conv import SAGEConv, GATv2Conv, GATConv, GCNConv, GCN2Conv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import OptTensor, Adj


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
        r"""The graph attentional operator from the `"Graph Attention Networks, paired with a normalization layer."
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
                    dimensionality.
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

    def forward(self, x, edge_index, edge_attr=None, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index, edge_attr=edge_attr)


class GCNConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, bias: bool = True, **kwargs):
        r"""The graph convolutional operator from the `"Semi-supervised
            Classification with Graph Convolutional Networks"
            <https://arxiv.org/abs/1609.02907>`_ paper, paired with a normalization layer.

            .. math::
                \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
                \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

            where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
            adjacency matrix with inserted self-loops and
            :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
            The adjacency matrix can include other values than :obj:`1` representing
            edge weights via the optional :obj:`use_edge_weight` tensor.

            Its node-wise formulation is given by:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
                \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
                \hat{d}_i}} \mathbf{x}_j

            with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
            :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
            node :obj:`i` (default: :obj:`1.0`)

            Args:
                in_channels (int): Size of each input sample, or :obj:`-1` to derive
                    the size from the first input(s) to the forward method.
                out_channels (int): Size of each output sample.
                improved (bool, optional): If set to :obj:`True`, the layer computes
                    :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
                    (default: :obj:`False`)
                cached (bool, optional): If set to :obj:`True`, the layer will cache
                    the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
                    \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
                    cached version for further executions.
                    This parameter should only be set to :obj:`True` in transductive
                    learning scenarios. (default: :obj:`False`)
                add_self_loops (bool, optional): If set to :obj:`False`, will not add
                    self-loops to the input graph. (default: :obj:`True`)
                normalize (bool, optional): Whether to add self-loops and compute
                    symmetric normalization coefficients on the fly.
                    (default: :obj:`True`)
                bias (bool, optional): If set to :obj:`False`, the layer will not learn
                    an additive bias. (default: :obj:`True`)
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.

            Shapes:
                - **input:**
                  node features :math:`(|\mathcal{V}|, F_{in})`,
                  edge indices :math:`(2, |\mathcal{E}|)`,
                  edge weights :math:`(|\mathcal{E}|)` *(optional)*
                - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
            """

        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)

        self.conv = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
            **kwargs
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index, edge_weight=edge_weight)


class GCN2ConvBlock(torch.nn.Module):
    def __init__(self, channels: int, alpha: float, theta: float = None, layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True, normalize: bool = True, **kwargs):
        r"""The graph convolutional operator with initial residual connections and
            identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
            Networks" <https://arxiv.org/abs/2007.02133>`_ paper, paired with a normalization layer.

            .. math::
                \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
                \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
                \mathbf{\Theta} \right)

            with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}`, where
            :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
            matrix with inserted self-loops and
            :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
            and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
            Here, :math:`\alpha` models the strength of the initial residual
            connection, while :math:`\beta` models the strength of the identity
            mapping.
            The adjacency matrix can include other values than :obj:`1` representing
            edge weights via the optional :obj:`use_edge_weight` tensor.

            Args:
                channels (int): Size of each input and output sample.
                alpha (float): The strength of the initial residual connection
                    :math:`\alpha`.
                theta (float, optional): The hyperparameter :math:`\theta` to compute
                    the strength of the identity mapping
                    :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
                    (default: :obj:`None`)
                layer (int, optional): The layer :math:`\ell` in which this module is
                    executed. (default: :obj:`None`)
                shared_weights (bool, optional): If set to :obj:`False`, will use
                    different weight matrices for the smoothed representation and the
                    initial residual ("GCNII*"). (default: :obj:`True`)
                cached (bool, optional): If set to :obj:`True`, the layer will cache
                    the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
                    \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
                    cached version for further executions.
                    This parameter should only be set to :obj:`True` in transductive
                    learning scenarios. (default: :obj:`False`)
                normalize (bool, optional): Whether to add self-loops and apply
                    symmetric normalization. (default: :obj:`True`)
                add_self_loops (bool, optional): If set to :obj:`False`, will not add
                    self-loops to the input graph. (default: :obj:`True`)
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.

            Shapes:
                - **input:**
                  node features :math:`(|\mathcal{V}|, F)`,
                  initial node features :math:`(|\mathcal{V}|, F)`,
                  edge indices :math:`(2, |\mathcal{E}|)`,
                  edge weights :math:`(|\mathcal{E}|)` *(optional)*
                - **output:** node features :math:`(|\mathcal{V}|, F)`
            """

        super().__init__()
        self.norm = LayerNorm(channels, elementwise_affine=True)

        self.conv = GCN2Conv(
            channels=channels,
            alpha=alpha,
            theta=theta,
            layer=layer,
            cached=cached,
            shared_weights=shared_weights,
            add_self_loops=add_self_loops,
            normalize=normalize,
            **kwargs
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index, edge_weight=edge_weight)
