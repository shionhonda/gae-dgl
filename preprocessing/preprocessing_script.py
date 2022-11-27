from torch_geometric.loader import DataLoader
from preprocessing.constants import PSCDB_PATH, PSCDB_CLEANED_TRAIN
from preprocessing.dataset import create_dataset_pscdb, load_dataset
from preprocessing.utils import pscdb_read


def main():
    df = pscdb_read(PSCDB_PATH)
    df2 = df.iloc[0:10]
    ds = create_dataset_pscdb(df2, export_path=PSCDB_CLEANED_TRAIN, in_memory=True)

    dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break

    """
    # This doesn't work
    ds2 = load_dataset(PSCDB_CLEANED_TRAIN)
    dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
    for el in dl:
        print(el)
        break
    """


if __name__ == '__main__':
    main()