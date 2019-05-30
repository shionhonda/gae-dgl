import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
     
    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
     
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        h =  g.ndata.pop('h')
        return h

# class GCN(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(GCN, self).__init__()
#         self.layer = GraphConv(in_feats, out_feats, activation=activation)

#     def forward(self, g, features):
#         h = self.layer(features, g)
#         return g, h

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super(GAE, self).__init__()
        layers = [GCN(in_dim, hidden_dims[0], F.relu)]
        if len(hidden_dims)>2:
            for i in range(1,len(hidden_dims)-1):
                layers.append(GCN(hidden_dims[i-1], hidden_dims[i], F.relu))
        layers.append(GCN(hidden_dims[-2], hidden_dims[-1], lambda x:x))
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

# class GAE(nn.Module):
#     def __init__(self, in_dim, hidden_dims, dropout=0.1):
#         super(GAE, self).__init__()
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GraphConv(in_dim, hidden_dims[0], activation=F.relu))
#         # output layer
#         for i in range(1,len(hidden_dims)):
#             self.layers.append(GraphConv(hidden_dims[i-1], hidden_dims[i]))
#         self.dropout = nn.Dropout(p=dropout)
#         self.decoder = InnerProductDecoder(activation=lambda x:x)

#     def forward(self, g, features):
#         h = features
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = layer(h, g)
#         adj_rec = self.decoder(h)
#         return adj_rec

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj