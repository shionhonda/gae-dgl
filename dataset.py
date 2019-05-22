import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dgl import DGLGraph
from rdkit import Chem

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
             'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 23 + degree, charge, is_aromatic = 35
BOND_FDIM = 5 + 6  # bond type


class ChemblDataset(Dataset):

    def __init__(self, corpus_path='chembl_24.csv', max_size=1000000):
        df = pd.read_csv(corpus_path)
        self.smiles = df['canonical_smiles'].values[:max_size]
        graphs, xs = [], []
        for sm in self.smiles:
            g, x, _ = mol2dgl_single(sm)
            graphs.append(g)
            xs.append(x)
        self.graphs = graphs
        self.xs = xs

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        return self.graphs[item], self.xs[item]

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return (torch.Tensor(fbond + fstereo))

def mol2dgl_single(sm):
    """
    in: single smiles string
    out: graph, atom_feature, bond_feature
    """
    n_edges = 0

    atom_x = []
    bond_x = []

    mol = get_mol(sm)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    graph = DGLGraph()
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        atom_x.append(atom_features(atom))
    graph.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = bond_features(bond)
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        # set up the reverse direction
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    graph.add_edges(bond_src, bond_dst)

    n_edges += n_bonds
    return graph, torch.stack(atom_x), \
            torch.stack(bond_x) if len(bond_x) > 0 else torch.zeros(0)