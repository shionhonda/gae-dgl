import os
import json
import pandas as pd
import torch
from graphein.protein import ProteinGraphConfig
from sklearn.preprocessing import LabelBinarizer
from preprocessing.constants import MOTION_TYPE, PDB, PARAMS_DIR_SUFFIX, PARAMS_CSV_SUFFIX, PARAMS_JSON_SUFFIX
from functools import partial
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_k_nn_edges, \
    add_ionic_interactions
from graphein.protein.features.nodes import amino_acid_one_hot, meiler_embedding, hydrogen_bond_donor, \
    expasy_protein_scale, hydrogen_bond_acceptor
from graphein.ml import InMemoryProteinGraphDataset, GraphFormatConvertor, ProteinGraphDataset
import graphein.ml.conversion as gmlc
from typing import final, Union, Optional, List, Any
from preprocessing.utils import FrozenDict


# Globally-visible constants
EDGE_CONSTRUCTION_FUNCTIONS: final = frozenset([
    partial(add_k_nn_edges, k=3, long_interaction_threshold=0),
    add_hydrogen_bond_interactions,
    add_peptide_bonds,
    # add_ionic_interactions
])
NODE_METADATA_FUNCTIONS: final = FrozenDict({
    # "amino_acid_one_hot": amino_acid_one_hot,
    "meiler": meiler_embedding,
    # "hbond_donors": hydrogen_bond_donor,
    # "hbond_acceptors": hydrogen_bond_acceptor,
    # "expasy": expasy_protein_scale
})
DATASET_NAME_PSCDB: final = "pscdb_cleaned"
DATASET_NAME_PRETRAINED: final = "pretrain_cleaned"
FORMATS: final = frozenset(["pyg", "dgl"])
VERBOSITIES_CONVERSION: final = frozenset(gmlc.SUPPORTED_VERBOSITY)
DATASET_TYPES: final = frozenset(["pscdb", "pretrain"])

# Local-only constants
__DATAFRAME_PARAM_NAME: final = "df_param_name"


def __load_params(path: str) -> dict[str, Any]:
    """
    It reads a csv file and a json file containing the parameters, and combines them into a single dictionary.

    :param path: str
    :type path: str
    :return: A dictionary of parameters.
    """
    # Read other parameters from json file
    with open(os.path.join(path, PARAMS_JSON_SUFFIX), "r") as fp:
        params = json.load(fp)

    if "df_param_name" in params:
        # Read param dataframe from csv
        df = pd.read_csv(os.path.join(path, PARAMS_CSV_SUFFIX))

        # Handle additional parameter for dataframe
        df_param_name = params[__DATAFRAME_PARAM_NAME]  # read parameter name
        del params[__DATAFRAME_PARAM_NAME]  # delete parameter name from parameter list
        params[df_param_name] = df  # add dataframe parameter corresponding to df_param_name
    return params


def __store_params(path: str, **kwargs):
    """
    It stores the given dataframe as a csv file and the given parameters as a json file

    :param path: str
    :type path: str
    :param df: the dataframe to store
    :type df: pd.DataFrame
    :param df_param_name: The name of the param eter that contains the dataframe
    :type df_param_name: str
    """
    # Store given dataframe as csv 
    params: dict = kwargs
    if "df" in params:
        df = params["df"]
        del params["df"]
        df_param_name = params["df_param_name"]
        df.to_csv(os.path.join(path, PARAMS_CSV_SUFFIX))
        # additional parameter for dataframe parameter name
        params.update({__DATAFRAME_PARAM_NAME: df_param_name})

    # Store other params as json
    with open(os.path.join(path, PARAMS_JSON_SUFFIX), "w") as fp:
        json.dump(params, fp)


def create_dataset_pscdb(df: pd.DataFrame, export_path: str, in_memory: bool = False, graph_format: str = "pyg",
                         conversion_verbosity: str = "gnn", store_params: bool = False) -> \
        Union[InMemoryProteinGraphDataset, ProteinGraphDataset]:
    """
    It takes a dataframe, extracts the PDB codes and the labels, creates a graphein config, a graph format converter and
    a dataset object.

    :param df: the dataframe containing the PDB codes and the labels
    :type df: pd.DataFrame
    :param export_path: The path to the directory where the dataset will be stored
    :type export_path: str
    :param in_memory: if True, the dataset will be loaded in memory. If False, the dataset will be loaded on-demand,
    defaults to False
    :type in_memory: bool (optional)
    :param graph_format: the format of the graph you want to store, defaults to pyg
    :type graph_format: str (optional)
    :param conversion_verbosity: This parameter controls the verbosity of the conversion process. It can be one of the
    following:, defaults to gnn
    :type conversion_verbosity: str (optional)
    :param store_params: bool = False, defaults to False
    :type store_params: bool (optional)
    :return: A dataset object
    """

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

    # Define graphein config, starting with the edge construction functions
    config = {
        "edge_construction_functions": list(EDGE_CONSTRUCTION_FUNCTIONS)
    }

    # Handle additional node features like Meiler's embeddings and amino-acid one-hot encoding, updating the config dict
    if len(NODE_METADATA_FUNCTIONS) > 0:
        config.update({"node_metadata_functions": list(NODE_METADATA_FUNCTIONS.values())})
    config = ProteinGraphConfig(**config)

    # Adding additional node features to the columns the graph format converter needs to store
    columns = list(NODE_METADATA_FUNCTIONS.keys())
    if conversion_verbosity == "gnn":
        columns.extend([
            "edge_index",
            "coords",
            "dist_mat",
            "name",
            "node_id",
        ])
    elif conversion_verbosity == "default":
        columns.extend([
            "b_factor",
            "chain_id",
            "coords",
            "dist_mat",
            "edge_index",
            "kind",
            "name",
            "node_id",
            "residue_name",
        ])
    elif conversion_verbosity == "all_info":
        columns.extend([
            "atom_type",
            "b_factor",
            "chain_id",
            "chain_ids",
            "config",
            "coords",
            "dist_mat",
            "edge_index",
            "element_symbol",
            "kind",
            "name",
            "node_id",
            "node_type",
            "pdb_df",
            "raw_pdb_df",
            "residue_name",
            "residue_number",
            "rgroup_df",
            "sequence_A",
            "sequence_B",
        ])

    # Format converter
    converter = GraphFormatConvertor(src_format="nx", dst_format=graph_format, columns=columns)

    # Create dataset
    if in_memory:
        ds = InMemoryProteinGraphDataset(
            root=export_path,
            name=DATASET_NAME_PSCDB,
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

    # Store given parameters if required
    if store_params:
        __store_params(
            path=os.path.join(export_path, PARAMS_DIR_SUFFIX),
            df=df,
            df_param_name="df",
            graph_format=graph_format,
            conversion_verbosity=conversion_verbosity,
            in_memory=in_memory
        )
    return ds


def create_dataset_pretrain(pdb_paths: List[str], export_path: str, in_memory: bool = False, graph_format: str = "pyg",
                            conversion_verbosity: str = "gnn", store_params: bool = False) -> \
        Union[InMemoryProteinGraphDataset, ProteinGraphDataset]:
    """
        This function takes in a list of pdb files, and returns a dataset of graphs

        :param pdb_paths: List[str]
        :type pdb_paths: List[str]
        :param export_path: The path to the directory where the dataset will be stored
        :type export_path: str
        :param in_memory: If True, the dataset will be loaded in memory. If False, the dataset will be loaded on disk,
            defaults to False
        :type in_memory: bool (optional)
        :param graph_format: The format of the graph. Can be one of the following:, defaults to pyg
        :type graph_format: str (optional)
        :param conversion_verbosity: str = "gnn", store_params: bool = False, defaults to gnn
        :type conversion_verbosity: str (optional)
        :param store_params: bool = False, defaults to False
        :type store_params: bool (optional)
        :return: A dataset object
    """

    if graph_format not in FORMATS:
        raise ValueError(f"Invalid graph format: {graph_format}, it needs to be one of the following: {str(FORMATS)}")

    if conversion_verbosity not in VERBOSITIES_CONVERSION:
        raise ValueError(f"Invalid conversion verbosity: {conversion_verbosity}, it needs to be one of the following: "
                         f"{str(VERBOSITIES_CONVERSION)}")
    '''
    # Extract PDBs
    if pdb_codes is not None:
        pdbs = pdb_codes
    else:
        pdbs = []
    pdbs = pdbs + pscdb[PDB].to_list()
    '''

    # Define graphein config
    config = {
        "edge_construction_functions": list(EDGE_CONSTRUCTION_FUNCTIONS)
    }

    # Handle additional node features like Meiler's embeddings and amino-acid one-hot encoding, updating the config dict
    if len(NODE_METADATA_FUNCTIONS) > 0:
        config.update({"node_metadata_functions": list(NODE_METADATA_FUNCTIONS.values())})
    config = ProteinGraphConfig(**config)

    # Adding additional node features to the columns the graph format converter needs to store
    columns = list(NODE_METADATA_FUNCTIONS.keys())
    if conversion_verbosity == "gnn":
        columns.extend([
            "edge_index",
            "coords",
            "dist_mat",
            "name",
            "node_id",
        ])
    elif conversion_verbosity == "default":
        columns.extend([
            "b_factor",
            "chain_id",
            "coords",
            "dist_mat",
            "edge_index",
            "kind",
            "name",
            "node_id",
            "residue_name",
        ])
    elif conversion_verbosity == "all_info":
        columns.extend([
            "atom_type",
            "b_factor",
            "chain_id",
            "chain_ids",
            "config",
            "coords",
            "dist_mat",
            "edge_index",
            "element_symbol",
            "kind",
            "name",
            "node_id",
            "node_type",
            "pdb_df",
            "raw_pdb_df",
            "residue_name",
            "residue_number",
            "rgroup_df",
            "sequence_A",
            "sequence_B",
        ])

    # Format converter
    converter = GraphFormatConvertor(src_format="nx", dst_format=graph_format, columns=columns)

    # Create dataset
    if in_memory:
        ds = InMemoryProteinGraphDataset(
            pdb_paths=pdb_paths,
            root=export_path,
            name=DATASET_NAME_PRETRAINED,
            graphein_config=config,
            graph_format_convertor=converter
        )
    else:
        ds = ProteinGraphDataset(
            pdb_paths=pdb_paths,
            root=export_path,
            graphein_config=config,
            graph_format_convertor=converter
        )

    # Store given parameters if required
    if store_params:
        __store_params(
            path=os.path.join(export_path, PARAMS_DIR_SUFFIX),
            pdb_paths=pdb_paths,
            graph_format=graph_format,
            conversion_verbosity=conversion_verbosity,
            in_memory=in_memory
        )

    return ds


def load_dataset(path: str, dataset_type: str = "pscdb") -> Union[InMemoryProteinGraphDataset, ProteinGraphDataset]:
    """
    Loads a protein graph cleaned dataset from a directory.

    :param path: The path to the dataset
    :type path: str
    :param dataset_type: type of dataset to load, either 'pscdb' or 'pretrain'
    :return: the ProteinGraphDataset or InMemoryProteinGraphDataset object corresponding to the dataset.
    """

    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Invalid dataset type '{dataset_type}', it must be one of: {DATASET_TYPES}")

    # Load parameters
    params = __load_params(os.path.join(path, PARAMS_DIR_SUFFIX))

    # Load dataset
    ds = None
    if dataset_type == "pscdb":
        ds = create_dataset_pscdb(export_path=path, **params)
    elif dataset_type == "pretrain":
        ds = create_dataset_pretrain(export_path=path, **params)

    return ds
