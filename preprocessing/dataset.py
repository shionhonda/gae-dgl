import pandas as pd
import torch
from graphein.protein import ProteinGraphConfig
from sklearn.preprocessing import LabelBinarizer
from preprocessing.constants import MOTION_TYPE, PDB
from functools import partial
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_k_nn_edges, \
    add_ionic_interactions
from graphein.ml import InMemoryProteinGraphDataset, GraphFormatConvertor, ProteinGraphDataset
import graphein.ml.conversion as gmlc
from typing import final, Union, Optional, List


EDGE_CONSTRUCTION_FUNCTIONS: final = frozenset([
    partial(add_k_nn_edges, k=3, long_interaction_threshold=0),
    add_hydrogen_bond_interactions,
    add_peptide_bonds,
    # add_ionic_interactions
])
DATASET_NAME: final = "pscdb_cleaned"
FORMATS: final = frozenset(["pyg", "dgl"])
VERBOSITIES_CONVERSION: final = frozenset(gmlc.SUPPORTED_VERBOSITY)


def create_dataset_pscdb(df: pd.DataFrame, export_path: str, in_memory: bool = False, graph_format: str = "pyg",
                         conversion_verbosity: str = "gnn") -> Union[InMemoryProteinGraphDataset, ProteinGraphDataset]:

    if graph_format not in FORMATS:
        raise ValueError(f"Invalid graph format: {graph_format}, it needs to be one of the following: {str(FORMATS)}")

    if conversion_verbosity not in VERBOSITIES_CONVERSION:
        raise ValueError(f"Invalid conversion verbosity: {conversion_verbosity}, it needs to be one of the following: "
                         f"{str(VERBOSITIES_CONVERSION)}")

    # Extract label
    one_hot_encode = LabelBinarizer().fit_transform(df[MOTION_TYPE])  # one hot encode labels
    y = [torch.argmax(torch.Tensor(lab)).type(torch.LongTensor) for lab in one_hot_encode]  # convert to sparse labels

    # Extract PDBs
    pdbs = df[PDB].to_list()

    # If dataset must be in-memory, create graph-level label map
    graph_label_map = {}
    if in_memory:
        for i in range(0, len(pdbs)):
            graph_label_map[pdbs[i]] = y[i]

    # Define graphein config
    config = {
        "edge_construction_functions": list(EDGE_CONSTRUCTION_FUNCTIONS)
    }
    config = ProteinGraphConfig(**config)

    # Format converter
    converter = GraphFormatConvertor(src_format="nx", dst_format=graph_format, verbose=conversion_verbosity)

    # Create dataset
    if in_memory:
        ds = InMemoryProteinGraphDataset(
            root=export_path,
            name=DATASET_NAME,
            pdb_codes=pdbs,
            graphein_config=config,
            graph_format_convertor=converter,
            graph_label_map=graph_label_map
        )
    else:
        ds = ProteinGraphDataset(
            root=export_path,
            pdb_codes=pdbs,
            graphein_config=config,
            graph_format_convertor=converter,
            graph_labels=y
        )
    return ds


def create_dataset_pretrain(pscdb: pd.DataFrame, export_path: str, uniprot_ids: Optional[List[str]] = None,
                            pdb_codes: Optional[List[str]] = None, in_memory: bool = False, graph_format: str = "pyg",
                            conversion_verbosity: str = "gnn") -> Union[InMemoryProteinGraphDataset,
                                                                        ProteinGraphDataset]:

    if graph_format not in FORMATS:
        raise ValueError(f"Invalid graph format: {graph_format}, it needs to be one of the following: {str(FORMATS)}")

    if conversion_verbosity not in VERBOSITIES_CONVERSION:
        raise ValueError(f"Invalid conversion verbosity: {conversion_verbosity}, it needs to be one of the following: "
                         f"{str(VERBOSITIES_CONVERSION)}")
    # Extract PDBs
    if pdb_codes is not None:
        pdbs = pdb_codes
    else:
        pdbs = []
    pdbs = pdbs + pscdb[PDB].to_list()

    # Define graphein config
    config = {
        "edge_construction_functions": list(EDGE_CONSTRUCTION_FUNCTIONS)
    }
    config = ProteinGraphConfig(**config)

    # Format converter
    converter = GraphFormatConvertor(src_format="nx", dst_format=graph_format, verbose=conversion_verbosity)

    # Create dataset
    if in_memory:
        ds = InMemoryProteinGraphDataset(
            root=export_path,
            name=DATASET_NAME,
            pdb_codes=pdbs,
            uniprot_ids=uniprot_ids,
            graphein_config=config,
            graph_format_convertor=converter
        )
    else:
        ds = ProteinGraphDataset(
            root=export_path,
            pdb_codes=pdbs,
            uniprot_ids=uniprot_ids,
            graphein_config=config,
            graph_format_convertor=converter
        )
    return ds


def load_dataset(path: str, in_memory_name: Optional[str] = None) -> Union[InMemoryProteinGraphDataset,
                                                                           ProteinGraphDataset]:
    """
    Loads a protein graph cleaned dataset from a directory.

    :param path: The path to the dataset
    :type path: str
    :param in_memory_name: If you want to load the dataset into memory, you can specify a name for it
    :type in_memory_name: Optional[str]
    :return: A dataset object
    """
    # TODO: fix this function, it doesn't work currently because of a weird PyTorch Geometric error
    if in_memory_name is None:
        ds = ProteinGraphDataset(
            root=path,
        )
    else:
        ds = InMemoryProteinGraphDataset(
            root=path,
            name=in_memory_name
        )
    return ds
