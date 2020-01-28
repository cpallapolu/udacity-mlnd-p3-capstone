
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from lstm_model import LSTMPredictor
from lstm_rsme_loss import RSMELoss


def model_fn(model_dir):
    print('Loading model.')

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print('model_info: {}'.format(model_info))

    # Determine the device and construct the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMPredictor(
        model_info['input_dim'],
        model_info['hidden_dim'],
        model_info['output_dim'],
        model_info['n_layers']
    )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print('Done loading model.')
    return model


def _get_data_loader(batch_size, zip_file_path):
    print('Get data loader from {}.'.format(zip_file_path))

    data_df = pd.read_csv(
        zip_file_path,
        dtype={'fullVisitorId': 'str'},
        compression='zip'
    )

    total_rows = data_df.shape[0] - (data_df.shape[0] % batch_size)

    data_y = data_df['totals.transactionRevenue'].values

    data_X = data_df.drop(
        ['totals.transactionRevenue', 'fullVisitorId'],
        axis=1
    ).values

    data_y = torch.from_numpy(data_y).float().squeeze()
    data_X = torch.from_numpy(data_X).float()

    data_X = data_X.reshape(data_X.shape[0], 1, data_X.shape[1])

    data_ds = TensorDataset(data_X, data_y)

    return DataLoader(data_ds, shuffle=True, batch_size=batch_size)


def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device):
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            counter += 1

            batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_X)

            loss = loss_fn(output.squeeze(), batch_y.float())
            loss.backward()

            optimizer.step()
            
            total_loss += loss.item()

#             if counter % 500 == 0:
#                 val_losses = []

#                 model.eval()

#                 for val in val_loader:
#                     val_X, val_y = val

#                     val_X = val_X.to(device)
#                     val_y = val_y.to(device)
#                       with torch.no_grad():
#                       val_output = model(val_X)

#                     val_loss = loss_fn(val_output.squeeze(), val_y.float())
#                     val_losses.append(val_loss.item())

#                 model.train()

        print(
            'Epoch: {}/{}...'.format(epoch, epochs),
#                     'Step: {}...'.format(counter),
            'Loss: {:.10f}...'.format(total_loss / len(train_loader))
#                     'Val Loss: {:.10f}'.format(np.mean(val_losses))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    SM_HOSTS = json.loads(os.environ['SM_HOSTS'])
    SM_CURRENT_HOST = os.environ['SM_CURRENT_HOST']
    SM_MODEL_DIR = os.environ['SM_MODEL_DIR']
    SM_CHANNEL_TRAIN = os.environ['SM_CHANNEL_TRAIN']
    SM_NUM_GPUS = os.environ['SM_NUM_GPUS']

    parser_args = [
        # Training Parameters
        {
            'name': '--batch-size',
            'type': int,
            'default': 1024,
            'metavar': 'N',
            'help': 'input batch size for training (default: 1024)'
        },
        {
            'name': '--epochs',
            'type': int,
            'default': 10,
            'metavar': 'N',
            'help': 'number of epochs to train (default: 10)'
        },
        {
            'name': '--seed',
            'type': int,
            'default': 1,
            'metavar': 'S',
            'help': 'random seed (default: 1)'
        },
        # Model Parameters
        {
            'name': '--input_dim',
            'type': int,
            'default': 32,
            'metavar': 'N',
            'help': 'size of input feature (default: 32)'
        },
        {
            'name': '--hidden_dim',
            'type': int,
            'default': 256,
            'metavar': 'N',
            'help': 'size of the hidden dimension (default: 256)'
        },
        {
            'name': '--output_dim',
            'type': int,
            'default': 1,
            'metavar': 'N',
            'help': 'size of the output dimension (default: 1)'
        },
        {
            'name': '--n_layers',
            'type': int,
            'default': 1,
            'metavar': 'N',
            'help': 'number of lstm layers (default: 1)'
        },
        # SageMaker Parameters
        {
            'name': '--hosts',
            'type': list,
            'default': SM_HOSTS,
            'metavar': 'H',
            'help': 'Environment hosts (default: {})'.format(SM_HOSTS)
        },
        {
            'name': '--current-host',
            'type': str,
            'default': SM_CURRENT_HOST,
            'metavar': 'C',
            'help': 'Environment current host (default: {})'.format(
                SM_CURRENT_HOST
            )
        },
        {
            'name': '--model-dir',
            'type': str,
            'default': SM_MODEL_DIR,
            'metavar': 'M',
            'help': 'Environment model dir (default: {})'.format(
                SM_MODEL_DIR
            )
        },
        {
            'name': '--data-dir',
            'type': str,
            'default': SM_CHANNEL_TRAIN,
            'metavar': 'C',
            'help': 'Environment channel train (default: {})'.format(
                SM_CHANNEL_TRAIN
            )
        },
        {
            'name': '--num-gpus',
            'type': int,
            'default': SM_NUM_GPUS,
            'metavar': 'N',
            'help': 'Environment number of gpus (default: {})'.format(
                SM_NUM_GPUS
            )
        },
    ]

    for parser_arg in parser_args:
        parser.add_argument(
            parser_arg['name'],
            type=parser_arg['type'],
            default=parser_arg['default'],
            metavar=parser_arg['metavar'],
            help=parser_arg['help']
        )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device {}.'.format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_data_loader(
        args.batch_size, os.path.join(args.data_dir, 'train.zip')
    )
    val_loader = _get_data_loader(
        args.batch_size, os.path.join(args.data_dir, 'val.zip')
    )

    # Build the model.
    model = LSTMPredictor(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.n_layers
    ).to(device)

    print(
        'Model loaded with input_dim {}, hidden_dim {}, outout_dim: {}, \
            n_layers {}.'.format(
              args.input_dim, args.hidden_dim, args.output_dim, args.n_layers))

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = RSMELoss()
    loss_fn = nn.MSELoss()

    train(
        model,
        train_loader,
        val_loader,
        args.epochs,
        optimizer,
        loss_fn,
        device
    )

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'n_layers': args.n_layers
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
