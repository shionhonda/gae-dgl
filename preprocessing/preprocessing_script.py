import os
from torch_geometric.loader import DataLoader
from preprocessing.constants import PSCDB_PATH, PSCDB_CLEANED_TRAIN, PRETRAIN_CLEANED_TRAIN, UNIPROTS_KEY, PDBS_KEY, \
    PATH_PDBS_JSON, PRETRAIN_CLEANED_VAL, PRETRAIN_CLEANED_TEST, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, \
    VAL_SIZE_PSCDB, TEST_SIZE_PSCDB, VAL_SIZE_PRETRAIN, TEST_SIZE_PRETRAIN, RANDOM_SEED, PSCDB_PDBS_SUFFIX
from preprocessing.dataset import create_dataset_pscdb, create_dataset_pretrain, load_dataset, DATASET_NAME_PSCDB, \
    DATASET_NAME_PRETRAINED
from preprocessing.utils import pscdb_read, get_uniprot_IDs_and_pdb_codes, train_test_validation_split, \
    get_pdb_paths_pscdb


def main():
    # Read raw data
    df = pscdb_read(PSCDB_PATH)
    df2 = df.iloc[0:10]

    # TODO: get pdb paths from json
    uniprots, pdbs, pdb_paths = get_uniprot_IDs_and_pdb_codes(PATH_PDBS_JSON)
    '''
    pdb_paths = [
        "data/alphafold/pdbs/AF-A8K5U8-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-A8K6D2-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-A8K9I1-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-B2R9Y1-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-B2RAL8-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-B4DHS0-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-B4DXQ7-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-B7Z7K3-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-Q7RM67-F1-model_v4.pdb",
        "data/alphafold/pdbs/AF-Q9Y499-F1-model_v4.pdb"
    ]  # TODO: remove this later
    '''

    # Perform train/validation/test split on PSCDB
    df_train, df_val, df_test = train_test_validation_split(
        df2,
        val_size    = VAL_SIZE_PSCDB,
        test_size   = TEST_SIZE_PSCDB,
        random_seed = RANDOM_SEED
    )

    # Perform train/validation/test split on pre-training proteins (PSCDB-ones excluded)
    pdb_paths_train, pdb_paths_val, pdb_paths_test = train_test_validation_split(
        pdb_paths,
        val_size    = VAL_SIZE_PRETRAIN,
        test_size   = TEST_SIZE_PRETRAIN,
        random_seed = RANDOM_SEED
    )

    # Get the PDB paths of the PSCDB train/validation/test proteins
    pscdb_pdb_paths_train = get_pdb_paths_pscdb(df_train, os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_val   = get_pdb_paths_pscdb(df_val,   os.path.join(PSCDB_CLEANED_VAL,   PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_test  = get_pdb_paths_pscdb(df_test,  os.path.join(PSCDB_CLEANED_TEST,  PSCDB_PDBS_SUFFIX))

    # Add PSCDB train/validation/test PDB paths to the respective pre-training PDB path lists
    pdb_paths_train = pdb_paths_train + pscdb_pdb_paths_train
    pdb_paths_val   = pdb_paths_val   + pscdb_pdb_paths_val
    pdb_paths_test  = pdb_paths_test  + pscdb_pdb_paths_test

    # Create PSCDB classification datasets
    ds_cl_train = create_dataset_pscdb(df_train, export_path=PSCDB_CLEANED_TRAIN, in_memory=True, store_params=True)
    ds_cl_val   = create_dataset_pscdb(df_val,   export_path=PSCDB_CLEANED_VAL,   in_memory=True, store_params=True)
    ds_cl_test  = create_dataset_pscdb(df_test,  export_path=PSCDB_CLEANED_TEST,  in_memory=True, store_params=True)

    # Create pre-training datasets
    # TODO: update these function calls according to the new parameters (path list)
    ds_pt_train = create_dataset_pretrain(
        pdb_paths=pdb_paths_train,
        export_path=PRETRAIN_CLEANED_TRAIN,
        in_memory=True,
        store_params=True
    )
    ds_pt_val = create_dataset_pretrain(
        pdb_paths=pdb_paths_val,
        export_path=PRETRAIN_CLEANED_VAL,
        in_memory=True,
        store_params=True
    )
    ds_pt_test = create_dataset_pretrain(
        pdb_paths=pdb_paths_test,
        export_path=PRETRAIN_CLEANED_TEST,
        in_memory=True,
        store_params=True
    )

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_pt_train, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break

    # Load the dataset and create the data loader to check if everything's ok
    ds2 = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    # ds2 = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break


if __name__ == '__main__':
    main()
