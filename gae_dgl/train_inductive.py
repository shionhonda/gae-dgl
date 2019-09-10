import argparse
import os

import dgl
import dill
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MolDataset
from gae import GAE


parser = argparse.ArgumentParser(description='Pre-train GAE')
parser.add_argument('--n_epochs', '-e', type=int, default=10, help='number of epochs')
parser.add_argument('--data_file', '-d', type=str, default='data/graphs.pkl', help='data file')
parser.add_argument('--save_dir', '-s', type=str, default='../result', help='result directry')
parser.add_argument('--in_dim', '-i', type=int, default=39, help='input dimension')
parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', help='list of hidden dimensions')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

def collate(samples):
    for g in samples:
        g.to(torch.device(device))
    bg = dgl.batch(samples)
    return bg

class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, g, train=True):
        adj = g.adjacency_matrix().to_dense().to(device)
        # alleviate imbalance
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        adj_logits = self.model.forward(g)
        loss = BCELoss(adj_logits, adj, pos_weight=pos_weight)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    
        return loss.item()

    def save(self, epoch, save_dir):
        output_path = os.path.join(save_dir, 'ep{:02}.pkl'.format(epoch))
        torch.save(self.model.state_dict(), output_path)

def plot(train_losses, val_losses):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()
    plt.save()

def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(save_dir, 'zinc250k.png'))

    model = GAE(args.in_dim, args.hidden_dims)
    model.to(device)
    #model.to('cuda:{}'.format(args.gpu_id))
    print('Loading data')
    with open(args.data_file, 'rb') as f:
        graphs = dill.load(f)
    print('Loaded {} molecules'.format(len(graphs)))
    train_graphs, val_graphs = train_test_split(graphs, test_size=10000)
    train_dataset = MolDataset(train_graphs)
    val_dataset = MolDataset(val_graphs)
    del train_graphs, val_graphs

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    trainer = Trainer(model, args)
    train_losses, val_losses = [], []
    print('Training Start')
    for epoch in tqdm(range(args.n_epochs)):
        train_loss = 0
        model.train()
        for bg in tqdm(train_loader):
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)  
            train_loss += trainer.iteration(bg)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        trainer.save(epoch, args.save_dir)

        val_loss = 0
        model.eval()
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