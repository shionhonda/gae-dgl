# GAE-DGL
Graph Auto-encoder [1] implemented with DGL by Shion Honda.  
Official implementation by the authors is [here](https://github.com/tkipf/gae) (TensorFlow, Python 2.7).

## Installation
### Prerequisites
You need PyTorch and DGL at least and the rest to try inductive settings with molecular graphs.

```
PyTorch
DeepGraphLibrary
RDKit
dill
tqdm
```

## Usage
### Transductive tasks (under development)
Reproduce the results of the paper [1] by the following command.

```
$ python train_transductive.py --dataset cora
```

You can switch the dataset to use by assigning to the `--dataset` option one from `cora/citeseer/pubmed`.

### Inductive tasks
This repository supports learning graph representations of molecules in the ZINC-250k dataset (or any unlabeled SMILES dataset). Run pre-training by the following commands.  

```
$ python prepare_data.py # download and preprocess zinc dataset
$ python train_inductive.py --hidden_dims 32 16 # pre-train GAE
```

The ZINC-250k is a subset of ZINC dataset and can be obtained easily by, for example, [Chainer Chemistry](https://github.com/pfnet-research/chainer-chemistry).  
Interestingly, I found GAE also works in inductive settings even though it was not tested in the original paper [1].

![](zinc250k.png)

## References
[1] Thomas N. Kipf and Max Welling. "[Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)." NIPS. 2016.