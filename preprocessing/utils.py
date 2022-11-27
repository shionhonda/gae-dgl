import json
import pandas as pd
from preprocessing.constants import UNIPROTS_KEY, PDBS_KEY, USED_COLUMNS, RANDOM_SEED, TEST_SIZE_PSCDB, VAL_SIZE_PSCDB
from sklearn.model_selection import train_test_split


def get_uniprot_IDs_and_pdb_codes(path: str) -> (list[str], list[str]):
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
        return uniprotIDs, pdbIDs


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


def train_test_validation_split(df: pd.DataFrame, val_size: float = VAL_SIZE_PSCDB, test_size: float = TEST_SIZE_PSCDB,
                                random_seed: int = RANDOM_SEED) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
   Splits a dataframe into train, validation and test sets.

   :param df: the dataframe to split
   :type df: pd.DataFrame
   :param val_size: the ratio of the validation set to the entire dataset
   :type val_size: float
   :param test_size: the ratio of the test set to the entire dataset
   :type test_size: float
   :param random_seed: The random seed to use for the split
   :type random_seed: int
   :return: A tuple of three dataframes.
   """
    df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_seed)
    df_train, df_test = train_test_split(df_train, test_size=val_size / (1 - test_size), random_state=random_seed)

    return df_train, df_val, df_test
