import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from dgl.data import register_data_args, load_data
from dgl import DGLGraph

from gae import GAE


class Trainer:
    def __init__(self, model, gpu_ids=[]):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, g, train=True):
        adj_rec = self.model.forward(g)
        loss = self.criterion(adj_rec, g.adjacency_matrix().to_dense())
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    
        return loss.item()

def collate(samples):
    graphs = list(samples)
    bg = dgl.batch(graphs)
    return bg

def main():
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    args = parser.parse_args()
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    
    print("""----Data statistics------'
      #Edges %d
      #Nodes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, data.graph.number_of_nodes(),
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))
    
    model = GAE(in_feats, [32,16])
    trainer = Trainer(model)
    n_epochs = 50
    losses = []
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        g = DGLGraph(data.graph)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        g.ndata['h'] = features
        loss = trainer.iteration(g)
        losses.append(loss)
        print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))
    
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.grid()
    plt.show()


if __name__=='__main__':
    main()