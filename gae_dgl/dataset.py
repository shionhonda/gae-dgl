from torch.utils.data import Dataset

class ChemblDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        print('Dataset inculudes {:d} graphs'.format(len(graphs)))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]