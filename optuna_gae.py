from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import optuna

from dataset import ChemblDataset
from gae import GAE

class Trainer:
    def __init__(self, model, optim, dataloader, gpu_ids=[]):
        self.model = model
        self.train_data = dataloader
        self.optim = optim
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

def get_trainer(trial, dataloader):
    n_layers = trial.suggest_categorical('n_layer', [2,3,4])
    hidden_dims = []
    for i in range(n_layers):
        hidden_dim = int(trial.suggest_loguniform('hidden_dim_{}'.format(i), 4, 256))
        hidden_dims.append(hidden_dim)
    model = GAE(39, hidden_dims)
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optim, dataloader)
    return trainer

def collate(samples):
    graphs = list(samples)
    bg = dgl.batch(graphs)
    return bg

def main():
    dataset = ChemblDataset(max_size=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

    def objective(trial):
        trainer = get_trainer(trial, dataloader)
        n_epochs = 30
        for epoch in tqdm(range(n_epochs)):
            for bg in dataloader:
                bg.set_e_initializer(dgl.init.zero_initializer)
                bg.set_n_initializer(dgl.init.zero_initializer)  
                loss = trainer.iteration(bg)
        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)


if __name__=='__main__':
    main()