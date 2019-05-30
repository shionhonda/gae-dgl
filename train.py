import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl

from dataset import ChemblDataset
from gae import GAE


def collate(samples):
    graphs = list(samples)
    bg = dgl.batch(graphs)
    return bg

def loss_function(logits, labels, pos_weight):
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_func(logits, labels)

class Trainer:
    def __init__(self, model, gpu_ids=[]):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, g, train=True):
        adj = g.adjacency_matrix().to_dense()
        pos_weight = torch.Tensor([(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        #pos_weight = torch.Tensor([1])
        adj_logits = self.model.forward(g)
        loss = loss_function(adj_logits, adj, pos_weight)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    
        return loss.item()

def main():
    model = GAE(39, [32,16])
    dataset = ChemblDataset(max_size=10000)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate)
    trainer = Trainer(model)
    n_epochs = 30
    losses = []
    it = 0
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        for bg in tqdm(dataloader):
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)  
            loss = trainer.iteration(bg)
            #if it%100 == 0:
            losses.append(loss)
            it += 1
        print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))
    
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.grid()
    plt.show()

# def main():
#     dataset = ChemblDataset(max_size=1000)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    
#     model = GAE(39, [32,16])
#     model.train()
#     optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#     n_epochs = 10
#     losses = []
#     it = 0
#     print('Training Start')
#     for epoch in tqdm(range(n_epochs)):
#         for bg in tqdm(dataloader):
#             bg.set_e_initializer(dgl.init.zero_initializer)
#             bg.set_n_initializer(dgl.init.zero_initializer)  
#             degs = bg.in_degrees().float()
#             norm = torch.pow(degs, -0.5)
#             norm[torch.isinf(norm)] = 0
#             bg.ndata['norm'] = norm.unsqueeze(1)
#             adj = bg.adjacency_matrix().to_dense()
#             #pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
#             pos_weight = torch.Tensor([1])
#             adj_logits = model.forward(bg, bg.ndata['h'])
#             loss = loss_function(adj_logits, adj, pos_weight=pos_weight)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             if it%100 == 0:
#                 losses.append(loss)
#             it += 1
#         print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))
#         print(torch.sigmoid(adj_logits))
    
#     plt.plot(losses)
#     plt.xlabel('iteration x100')
#     plt.ylabel('train loss')
#     plt.grid()
#     plt.show()


if __name__=='__main__':
    main()