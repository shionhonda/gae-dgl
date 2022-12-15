from torch_geometric.loader import DataLoader
from preprocessing.constants import PSCDB_PATH, PSCDB_CLEANED_TRAIN, PRETRAIN_CLEANED_TRAIN, UNIPROTS_KEY, PDBS_KEY, \
    PATH_PDBS_JSON
from preprocessing.dataset import create_dataset_pscdb, create_dataset_pretrain, load_dataset, DATASET_NAME_PSCDB, \
    DATASET_NAME_PRETRAINED
from preprocessing.utils import pscdb_read, get_uniprot_IDs_and_pdb_codes


def main():

    # Read raw data
    df = pscdb_read(PSCDB_PATH)
    df2 = df.iloc[0:6]
    uniprots, pdbs = get_uniprot_IDs_and_pdb_codes(PATH_PDBS_JSON)

    # Create dataset
    # ds = create_dataset_pscdb(df2, export_path=PSCDB_CLEANED_TRAIN, in_memory=True, store_params=True)
    ds = create_dataset_pretrain(
        df2,
        export_path=PRETRAIN_CLEANED_TRAIN,
        uniprot_ids=uniprots,
        pdb_codes=pdbs,
        in_memory=True,
        store_params=True
    )

    # Create data loader to test if everything's ok
    dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break

    # This doesn't work
    ds2 = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    # ds2 = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break


if __name__ == '__main__':
    main()
