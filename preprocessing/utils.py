import json
import pandas as pd
from preprocessing.constants import UNIPROTS_KEY, PDBS_KEY, USED_COLUMNS


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


def pcsdb_read(path: str):
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


