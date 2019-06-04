import dill
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import RDConfig

class ChemblDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        print('Dataset inculudes {:d} graphs'.format(len(graphs)))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]