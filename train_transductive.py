import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.data import register_data_args, load_data
from dgl import DGLGraph

from gae import GAE

def loss_function(logits, labels, pos_weight):
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

def main():
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    args = parser.parse_args()
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    in_feats = features.shape[1]
    
    model = GAE(in_feats, [32,16])
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    n_epochs = 200
    losses = []
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        g = DGLGraph(data.graph)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        adj = g.adjacency_matrix().to_dense()
        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        model.train()
        adj_logits = model.forward(g, features)
        loss = loss_function(adj_logits, adj, pos_weight=pos_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))
        print(torch.sigmoid(adj_logits))
    
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.grid()
    plt.show()


if __name__=='__main__':
    main()