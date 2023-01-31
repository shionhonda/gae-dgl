from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch.nn import LayerNorm
from torch_geometric.nn.models import DeepGraphInfomax
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def readout_function():
    # TODO must implement summary_function
    pass


def random_sample_corruption(training_set: DataLoader, graph: Data) -> Data:
    """
    Takes a DataLoader and returns a single random sample from it.

    :param training_set: DataLoader
    :type training_set: DataLoader
    :return: A single sample from the training set.
    """
    corrupted_graph = graph
    while corrupted_graph.edge_index == graph.edge_index:
        train_sample = RandomSampler(
            training_set,
            replacement=False,
            num_samples=1,
            generator=None
        )
        corrupted_graph = next(iter(train_sample))
    return corrupted_graph


class DeepGraphInfomaxWrapper(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 encoder: torch.nn.Module,
                 normalize_hidden: bool = True,
                 readout: Callable = None,
                 corruption: Callable = None,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)

        self.dgi = DeepGraphInfomax(
            hidden_channels,
            encoder,
            readout,
            corruption
        )

        self.__in_channels = in_channels
        self.__hidden_channels = hidden_channels
        self.__out_channels = out_channels
        self.__normalize_hidden = normalize_hidden
        self.dropout = dropout

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi(x, edge_index)

        if self.norm is not None:
            summary = self.norm(summary).relu()
        else:
            summary = F.relu(summary)

        # Apply dropout
        summary = F.dropout(summary, p=self.dropout, training=self.training)
        # Apply second projection if required
        if self.lin2 is not None:
            summary = self.lin2(summary)

        return summary


def train_step_D(model: VGAEv2, train_data: DataLoader, optimizer, device, use_edge_weight: bool = False,
                    use_edge_attr: bool = False):
    model.train()  # put the model in training mode

    running_loss = 0.0  # running average loss over the batches
    steps: int = 1

    for data in iter(train_data):
        data = data.to(device)  # move batch to device
        optimizer.zero_grad()  # reset the optimizer gradients

        # Encoder output
        if use_edge_weight and use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight)
        elif use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            z = model.encode(data.x, data.edge_index)

        loss = model.recon_loss(z, data.edge_index)  # reconstruction loss
        kl_divergence_loss = (1 / data.num_nodes) * model.kl_loss()  # KL-divergence loss, should work as mean on nodes
        loss = loss + kl_divergence_loss

        loss.backward()  # gradient update
        optimizer.step()  # advance the optimizer state

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_vgae(model: VGAEv2, val_data: DataLoader, device, use_edge_weight: bool = False,
                   use_edge_attr: bool = False):
    model.eval()  # put the model in evaluation mode

    # Running average for loss, precision and AUC
    running_val_loss = 0.0
    running_auc = 0.0
    running_precision = 0.0
    steps: int = 1

    for data in iter(val_data):
        data = data.to(device)  # move batch to device

        # Encoder output
        if use_edge_weight and use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight)
        elif use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            z = model.encode(data.x, data.edge_index)

        loss = model.recon_loss(z, data.edge_index)  # reconstruction loss
        kl_divergence_loss = (1 / data.num_nodes) * model.kl_loss()  # KL-divergence loss, should work as mean on nodes
        loss = loss + kl_divergence_loss
        running_val_loss = running_val_loss + 1 / steps * (
                loss.item() - running_val_loss)  # update loss running average

        # Update AUC and precision running averages
        neg_edge_index = negative_sampling(data.edge_index, z.size(0))
        auc, avg_precision = model.test(z, pos_edge_index=data.edge_index, neg_edge_index=neg_edge_index)
        running_auc = running_auc + 1 / steps * (auc - running_auc)
        running_precision = running_precision + 1 / steps * (avg_precision - running_precision)
        steps += 1

    return float(running_val_loss), running_auc, running_precision


def train_vgae(model: VGAEv2, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
               experiment_path: str, experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False,
               early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0) -> torch.nn.Module:
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Instantiate the summary writer
    writer = SummaryWriter(f'{experiment_path}_{experiment_name}_{epochs}_epochs')

    # Early-stopping monitor
    checkpoint_path = os.path.join(f"{experiment_path}", "checkpoint.pt")
    monitor = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True,
        delta=early_stopping_delta,
        path=checkpoint_path
    )

    # Metric history trace object
    mht = MetricsHistoryTracer(
        metrics=['train_loss', 'val_loss', 'auc_val', 'avg_precision_val'],
        name="VGAE training metrics"
    )

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_vgae(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        # Do validation step
        val_loss, auc, avg_precision = test_step_vgae(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        print('Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, '
              'AUC: {:.4f}, Average precision: {:.4f}'.format(epoch, train_loss, val_loss, auc, avg_precision))

        # Tensorboard state update
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('auc_val', auc, epoch)  # new line
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('avg_precision_val', avg_precision, epoch)  # new line

        # Check for early-stopping stuff
        monitor(val_loss, model)
        if monitor.early_stop:
            print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            break

        # Metrics history update
        mht.add_scalar('train_loss', train_loss)
        mht.add_scalar('val_loss', val_loss)
        mht.add_scalar('auc_val', auc)
        mht.add_scalar('avg_precision_val', avg_precision)

    # Plot the metrics
    mht.plot_metrics(
        ['train_loss', 'val_loss'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='val_loss',
        store_path=os.path.join(f"{experiment_path}", "loss.svg")
    )

    mht.plot_metrics(
        ['auc_val'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='auc_val',
        store_path=os.path.join(f"{experiment_path}", "auc.svg")
    )

    mht.plot_metrics(
        ['avg_precision_val'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='avg_precision_val',
        store_path=os.path.join(f"{experiment_path}", "avg_precision.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model
