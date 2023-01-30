from typing import Optional
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.autoencoder import GAE
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import negative_sampling


class GAEv2(GAE):
    def __init__(self, encoder: torch.nn.Module, decoder: Optional[torch.nn.Module] = None):
        """
        GAE sub-class with a simple forward function implemented.

        :param encoder: The encoder network
        :type encoder: torch.nn.Module
        :param decoder: Optional[torch.nn.Module] = None
        :type decoder: Optional[torch.nn.Module]
        """
        super(GAEv2, self).__init__(encoder=encoder, decoder=decoder)

    def forward(self, x, edge_index, sigmoid: bool = True, *args, **kwargs):
        z = self.encode(x, edge_index, *args, **kwargs)
        adj_rec = self.decode(z, sigmoid=sigmoid)
        return adj_rec


def train_step_gae(model: GAEv2, train_data: DataLoader, optimizer, device, use_edge_weight: bool = False,
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

        loss = model.recon_loss(z, data.edge_index)  # reconstruction

        loss.backward()  # gradient update
        optimizer.step()  # advance the optimizer state

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_gae(model: GAEv2, val_data: DataLoader, device, use_edge_weight: bool = False,
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
        running_val_loss = running_val_loss + 1/steps * (loss.item() - running_val_loss)  # update loss running average

        # Update AUC and precision running averages
        neg_edge_index = negative_sampling(data.edge_index, z.size(0))
        auc, avg_precision = model.test(z, pos_edge_index=data.edge_index, neg_edge_index=neg_edge_index)
        running_auc = running_auc + 1/steps * (auc - running_auc)
        running_precision = running_precision + 1/steps * (avg_precision - running_precision)
        steps += 1

    return float(running_val_loss), running_auc, running_precision


def train_gae(model: GAEv2, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer, experiment_path: str,
              experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False):

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Instantiate the summary writer
    writer = SummaryWriter(f'{experiment_path}_{experiment_name}_{epochs}_epochs')

    for epoch in range(0, epochs):

        # Do train step
        train_loss = train_step_gae(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        # Do validation step
        val_loss, auc, avg_precision = test_step_gae(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )  # validation metrics

        print('Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, '
              'AUC: {:.4f}, Average precision: {:.4f}'.format(epoch, train_loss, val_loss, auc, avg_precision))

        # Update tensorboard state
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('auc_val', auc, epoch)  # new line
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('avg_precision_val', avg_precision, epoch)  # new line
