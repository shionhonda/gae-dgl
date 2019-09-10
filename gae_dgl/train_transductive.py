import argparse
import os

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dill
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from gae import GAE

parser = argparse.ArgumentParser(description='Pre-train GAE')
register_data_args(parser)
parser.add_argument('--n_epochs', '-e', type=int, default=10, help='number of epochs')
parser.add_argument('--save_dir', '-s', type=str, default='../result', help='result directry')
parser.add_argument('--in_dim', '-i', type=int, default=39, help='input dimension')
parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', help='list of hidden dimensions')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='Adam learning rate')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # TODO: train test split
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    in_feats = features.shape[1]
    
    model = GAE(in_feats, [32,16])
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    g = DGLGraph(data.graph)
    g.ndata['h']


    n_epochs = 500
    losses = []
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        adj = g.adjacency_matrix().to_dense()
        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        
        
        adj_logits = model.forward(g, features)
        print(torch.sigmoid(adj_logits))
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