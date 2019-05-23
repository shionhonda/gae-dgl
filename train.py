from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

from dataset import ChemblDataset
from gae import GAE

class Trainer:
    def __init__(self, model, dataloader, gpu_ids=[]):
        self.model = model
        self.train_data = dataloader
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-6)
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
    model = GAE(39, [256,256])
    dataset = ChemblDataset(max_size=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    trainer = Trainer(model, dataloader)
    n_epochs = 100
    print('Training Start')
    for epoch in tqdm(range(n_epochs)):
        for bg in tqdm(dataloader):
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)  
            loss = trainer.iteration(bg)
        print('Epoch: {:02d} | Loss: {:.5f}'.format(epoch, loss))


if __name__=='__main__':
    main()