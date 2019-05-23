import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

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

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2):
        super(GAE, self).__init__()
        self.layers = nn.ModuleList([GCN(in_dim, hidden_dim_1, F.relu),
                                    GCN(hidden_dim_1, hidden_dim_2, F.relu)])
    
    def forward(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        a = torch.sigmoid(torch.matmul(h, torch.transpose(h, 1, 0)))
        return a