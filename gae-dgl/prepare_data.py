import argparse
import pandas as pd
import dill
from tqdm import tqdm
import torch
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import RDConfig

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
             'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 23 + degree, charge, is_aromatic = 39


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

def mol2dgl_single(mols):
    """
    inputs
      mols: a list of molecules
    outputs
      cand_graphs: a list of dgl graphs 
    """
    cand_graphs = []
    print('Transforming molecules to DGLGraphs')
    for mol in tqdm(mols):
        n_atoms = mol.GetNumAtoms()
        g = DGLGraph()        
        node_feats = []
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            node_feats.append(atom_features(atom))
        g.add_nodes(n_atoms)
        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
        g.add_edges(bond_src, bond_dst)
        
        g.ndata['h'] = torch.Tensor([a.tolist() for a in node_feats])
        cand_graphs.append(g)
    return cand_graphs

def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--save_dir', '-s', type=str, default='data', help='result directry')
    parser.add_argument('--data_file', '-d', type=str, default='chembl_24.csv', help='data file (.csv)')
    args = parser.parse_args()

    print('Loading data')
    smiles = pd.read_csv(args.save_dir + '/' + args.data_file)['canonical_smiles'].values
    mols = []
    print('Transforming SMILES to molecules')
    for sm in tqdm(smiles):
        mol = get_mol(sm)
        if mol is not None:
            mols.append(mol)
        else:
            print('Could not construct a molecule:', sm)
    with open(args.save_dir + '/mols.pkl', 'wb') as f:
        dill.dump(mols, f)
    graphs = mol2dgl_single(mols)
    del mols
    with open(args.save_dir + '/graphs.pkl', 'wb') as f:
        dill.dump(graphs, f)
    print('{:d} molecules constructed'.format(len(graphs)))


if __name__=='__main__':
    main()