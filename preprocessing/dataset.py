import torch
from torch_geometric.data import InMemoryDataset, Data
from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphListDataset
from typing import List, Tuple, Optional, Any


def pair_data(a: Data, b: Data) -> Data:
    """Pairs two graphs together in a single ``Data`` instance.

    The first graph is accessed via ``data.a`` (e.g. ``data.a.coords``) and the second via ``data.b``.

    :param a: The first graph.
    :type a: torch_geometric.data.Data
    :param b: The second graph.
    :type b: torch_geometric.data.Data
    :return: The paired graph.
    """
    out = Data()
    out.a = a
    out.b = b
    return out


class PairedProteinGraphListDataset(InMemoryDataset):
    def __init__(
            self, root: str, data_list: List[Tuple[Data, Data]], name: str, labels: Optional[Any] = None, transform=None
    ):
        """Creates a dataset from a list of PyTorch Geometric Data objects.
        :param root: Root directory where the dataset is stored.
        :type root: str
        :param data_list: List of protein graphs as PyTorch Geometric Data
            objects.
        :type data_list: List[Data]
        :param name: Name of dataset. Data will be saved as ``data_{name}.pt``.
        :type name: str
        :param transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        :type transform: Optional[Callable], optional
        """
        self.data_list = data_list
        self.name = name
        self.labels = labels
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        """The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return f"data_{self.name}.pt"

    def process(self):
        """Saves data files to disk."""
        # Pair data objects
        paired_data = [pair_data(a, b) for a, b in self.data_list]

        # Assign labels
        if self.labels is not None:
            for i, d in enumerate(paired_data):
                d.y = self.labels[i]

        torch.save(self.collate(paired_data), self.processed_paths[0])