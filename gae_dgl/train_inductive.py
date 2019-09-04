import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dill
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
    def __init__(self, model, args):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if self.device == 'cuda':
        #     self.model = nn.DataParallel(self.model, args.gpu_ids)
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, g, train=True):
        adj = g.adjacency_matrix().to_dense()
        pos_weight = torch.Tensor([(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        adj_logits = self.model.forward(g)
        loss = loss_function(adj_logits, adj, pos_weight)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    
        return loss.item()

    def save(self, epoch, save_dir):
        output_path = save_dir + '/ep{:02}.pkl'.format(epoch)
        torch.save(self.model.state_dict(), output_path)
        #self.model.to(self.device)

def plot(train_losses, val_losses):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--data_file', '-d', type=str, default='data/graphs.pkl', help='data file')
    parser.add_argument('--save_dir', '-s', type=str, default='../result', help='result directry')
    parser.add_argument('--in_dim', '-i', type=int, default=39, help='input dimension')
    parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', help='list of hidden dimensions')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = GAE(args.in_dim, [32,16])
    print('Loading data')
    with open(args.data_file, 'rb') as f:
        graphs = dill.load(f)
    print(len(graphs))
    train_graphs, val_graphs = train_test_split(graphs, test_size=10000)
    train_dataset = ChemblDataset(train_graphs)
    val_dataset = ChemblDataset(val_graphs)
    del train_graphs, val_graphs
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    trainer = Trainer(model, args)
    train_losses, val_losses = [], []
    print('Training Start')
    for epoch in tqdm(range(args.n_epochs)):
        train_loss = 0
        for bg in tqdm(train_loader):
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)  
            train_loss += trainer.iteration(bg)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        trainer.save(epoch, args.save_dir)
        val_loss = 0
        for bg in val_loader:
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)  
            val_loss += trainer.iteration(bg, train=False)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print('Epoch: {:02d} | Train Loss: {:.4f} | Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))
    plot(train_losses, val_losses)


if __name__=='__main__':
    main()