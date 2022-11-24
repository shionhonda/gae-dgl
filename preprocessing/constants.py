from typing import final
import os


# Generic dataset-related constants
DATA_PATH: final = "data"

# Additional PDBs from AlphaFold-related constants
UNIPROTS_KEY: final = "uniprotIDs"
PDBS_KEY: final = "pdbIDs"
PATH_PDBS: final = os.path.join(DATA_PATH, "alphafold", "pdbs.json")

# PSCDB-related constants
USED_COLUMNS: final = {"Free PDB": "pdb", "motion_type": "motion"}
PSCDB_PATH: final = os.path.join(DATA_PATH, "pscdb", "structural_rearrangement_data.csv")

# Randomness-related constants
RANDOM_SEED: final = 42
