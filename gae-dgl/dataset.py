import dill
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import RDConfig

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
             'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 23 + degree, charge, is_aromatic = 39


class ChemblDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        print('Dataset inculudes {:d} graphs'.format(len(graphs)))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]