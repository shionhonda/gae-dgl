import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

from dataset import ChemblDataset
from gae import GAE

def collate(samples):
    graphs = list(samples)
    bg = dgl.batch(graphs)
    return bg

def main():
    dataset = ChemblDataset(max_size=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

    model = GAE(39, 16, 16)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    epoch_losses = []
    for epoch in range(1000):
        epoch_loss = 0
        for i, bg in enumerate(dataloader):
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)        
            pred = model(bg)
            loss = loss_func(pred, bg.adjacency_matrix().to_dense())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (i + 1)
        if (epoch+1) % 1 == 0:
            print('Epoch {}, loss {:.4f}'.format(epoch+1, epoch_loss))
            print(pred)
        epoch_losses.append(epoch_loss)


if __name__=='__main__':
    main()