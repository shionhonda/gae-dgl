import os
import shutil
from torch_geometric.loader import DataLoader
from preprocessing.constants import PSCDB_PATH, PSCDB_CLEANED_TRAIN, PRETRAIN_CLEANED_TRAIN, PATH_PDBS_JSON, \
    PRETRAIN_CLEANED_VAL, PRETRAIN_CLEANED_TEST, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, \
    VAL_SIZE_PSCDB, TEST_SIZE_PSCDB, VAL_SIZE_PRETRAIN, TEST_SIZE_PRETRAIN, RANDOM_SEED, PSCDB_PDBS_SUFFIX, \
    PATH_PDBS_DIR
from preprocessing.dataset import create_dataset_pscdb, create_dataset_pretrain, load_dataset
from preprocessing.utils import pscdb_read, get_uniprot_IDs_and_pdb_codes, train_test_validation_split, \
    get_pdb_paths_pscdb


def main():
    # Read raw data
    df = pscdb_read(PSCDB_PATH, drop_duplicate_pdb_codes=True)
    df2 = df.iloc[0:-1]

    uniprots, pdbs, pdb_paths = get_uniprot_IDs_and_pdb_codes(PATH_PDBS_JSON)

    # Perform train/validation/test split on PSCDB
    df_train, df_val, df_test = train_test_validation_split(
        df2,
        val_size=VAL_SIZE_PSCDB,
        test_size=TEST_SIZE_PSCDB,
        random_seed=RANDOM_SEED
    )

    # Perform train/validation/test split on pre-training proteins (PSCDB-ones excluded)
    pdb_paths_train, pdb_paths_val, pdb_paths_test = train_test_validation_split(
        pdb_paths,
        val_size=VAL_SIZE_PRETRAIN,
        test_size=TEST_SIZE_PRETRAIN,
        random_seed=RANDOM_SEED
    )

    # Get the PDB paths of the PSCDB train/validation/test proteins
    '''
    pscdb_pdb_paths_train = get_pdb_paths_pscdb(df_train, os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_val = get_pdb_paths_pscdb(df_val, os.path.join(PSCDB_CLEANED_VAL, PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_test = get_pdb_paths_pscdb(df_test, os.path.join(PSCDB_CLEANED_TEST, PSCDB_PDBS_SUFFIX))
    '''
    pscdb_pdb_paths_train = get_pdb_paths_pscdb(df_train, PATH_PDBS_DIR)
    pscdb_pdb_paths_val = get_pdb_paths_pscdb(df_val, PATH_PDBS_DIR)
    pscdb_pdb_paths_test = get_pdb_paths_pscdb(df_test, PATH_PDBS_DIR)

    # Add PSCDB train/validation/test PDB paths to the respective pre-training PDB path lists
    pdb_paths_train = pdb_paths_train + pscdb_pdb_paths_train
    pdb_paths_val = pdb_paths_val + pscdb_pdb_paths_val
    pdb_paths_test = pdb_paths_test + pscdb_pdb_paths_test

    # Create PSCDB classification datasets
    ds_cl_train = create_dataset_pscdb(df_train, export_path=PSCDB_CLEANED_TRAIN, in_memory=True, store_params=True)
    ds_cl_val = create_dataset_pscdb(df_val, export_path=PSCDB_CLEANED_VAL, in_memory=True, store_params=True)
    ds_cl_test = create_dataset_pscdb(df_test, export_path=PSCDB_CLEANED_TEST, in_memory=True, store_params=True)

    # Copy PSCDB .pdb files to AlphaFold directory, otherwise pre-train dataset creation wont work cuz: "graphein cool!"
    copy_all_pscdb_files = input("Copy all PSCDB .pdb files to alphafold directory (0: no, 1: yes)? ")
    if int(copy_all_pscdb_files) != 0:
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR)
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_VAL, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR)
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_TEST, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR)

    # Create pre-training datasets
    ds_pt_train = create_dataset_pretrain(
        pdb_paths=pdb_paths_train,
        export_path=PRETRAIN_CLEANED_TRAIN,
        in_memory=False,
        store_params=True
    )
    ds_pt_val = create_dataset_pretrain(
        pdb_paths=pdb_paths_val,
        export_path=PRETRAIN_CLEANED_VAL,
        in_memory=False,
        store_params=True
    )
    ds_pt_test = create_dataset_pretrain(
        pdb_paths=pdb_paths_test,
        export_path=PRETRAIN_CLEANED_TEST,
        in_memory=False,
        store_params=True
    )

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_pt_train, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_pt_val, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_pt_test, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_train, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_val, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_test, batch_size=2, shuffle=True, drop_last=True)
    print(len(dl))
    print(next(iter(dl)))

    # Load the dataset and create the data loader to check if everything's ok
    ds2 = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    print(len(ds2))
    dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
    print(next(iter(dl)))

    ds3 = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    dl = DataLoader(ds3, batch_size=2, shuffle=True, drop_last=True)
    print(next(iter(dl)))


if __name__ == '__main__':
    main()
