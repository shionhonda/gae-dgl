import collections
import json
import os
from typing import Union, List
import pandas as pd
from preprocessing.constants import UNIPROTS_KEY, PDBS_KEY, USED_COLUMNS, RANDOM_SEED, TEST_SIZE_PSCDB, VAL_SIZE_PSCDB, \
    PDB, PATHS_KEY
from sklearn.model_selection import train_test_split


def get_uniprot_IDs_and_pdb_codes(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    This function takes in a path to a JSON file containing the UniProt IDs and PDB codes, reading them from the JSON
    file and returning them.

    :param path: the path to the json file containing the UniProt IDs and PDB codes.
    :type path: str
    :return: a tuple of two lists containing the UniProt IDs and PDB codes.
    """

    with open(path, "r") as fp:
        data = json.load(fp)
        uniprotIDs = data[UNIPROTS_KEY]
        pdbIDs = data[PDBS_KEY]
        paths = data[PATHS_KEY]
        return uniprotIDs, pdbIDs, paths


def pscdb_read(path: str):
    """
    It reads the CSV file at the given path, drops all columns except the ones we want, and renames the columns to the
    names we want.

    :param path: the path to the csv file
    :type path: str
    :return: A dataframe with the columns we want.
    """
    df = pd.read_csv(path)
    df = df.drop(df.columns.difference(USED_COLUMNS.keys()), axis=1)
    df = df.rename(columns=USED_COLUMNS)
    return df


def get_pdb_paths_pscdb(pscdb: pd.DataFrame, root_path: str) -> List[str]:
    """
    Given a PSCDB dataframe and a root path, return a list of paths to the PDB files.

    :param pscdb: the dataframe containing the PSCDB data
    :type pscdb: pd.DataFrame
    :param root_path: the path to the directory containing the PDB files
    :type root_path: str
    :return: A list of paths to the PDB files.
    """
    pdb_paths = pscdb[PDB].to_list()
    for i in range(0, len(pdb_paths)):
        pdb_paths[i] = os.path.join(root_path, pdb_paths[i] + ".pdb")
    return pdb_paths


def train_test_validation_split(dataset: Union[pd.DataFrame, List[str]], val_size: float = VAL_SIZE_PSCDB,
                                test_size: float = TEST_SIZE_PSCDB, random_seed: int = RANDOM_SEED) -> \
        tuple[Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]]]:
    """
    Splits a dataframe into train, validation and test sets.

    :param dataset: the dataframe to split
    :type dataset: pd.DataFrame
    :param val_size: the ratio of the validation set to the entire dataset
    :type val_size: float
    :param test_size: the ratio of the test set to the entire dataset
    :type test_size: float
    :param random_seed: The random seed to use for the split
    :type random_seed: int
    :return: A tuple of three dataframes.
    """

    if type(dataset) == list:
        df = pd.DataFrame(dataset)
    else:
        df = dataset

    df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_seed)
    df_train, df_test = train_test_split(df_train, test_size=val_size / (1 - test_size), random_state=random_seed)

    if type(dataset) == list:
        return df_train[0].to_list(), df_val[0].to_list(), df_test[0].to_list()
    else:
        return df_train, df_val, df_test


class FrozenDict(collections.Mapping):
    """
    Simple class representing a dictionary that cannot be changed.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs a new frozen dictionary from the given arguments.
        """
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __str__(self):
        return str(self._d)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash
