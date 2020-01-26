
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

from lstm_model import LSTMPredictor


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
        model_info['hidden_dim']
        )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print('Done loading model.')
    return model


def _get_train_data_loader(batch_size, training_dir):
    print('Get train data loader.')

    train_data = pd.read_csv(
        os.path.join(training_dir, 'train.zip'),
        dtype={'fullVisitorId': 'str'},
        compression='zip'
    )

    train_y = np.log1p(train_data['totals.transactionRevenue'].values)

    train_X = train_data.drop(
        ['totals.transactionRevenue', 'fullVisitorId'],
        axis=1
    ).values

    train_y = torch.from_numpy(train_y).float().squeeze()
    train_X = torch.from_numpy(train_X).float()

    train_X = train_X.reshape(765707, 1, 24)

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    
    return torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device, every_num=10):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_X)

            loss = loss_fn(output.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            
            print('Epoch: {}, BCELoss: {}'.format(epoch, total_loss / len(train_loader)))
#         if every_num % 10 == 0:



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
            'default': 512,
            'metavar': 'N',
            'help': 'input batch size for training (default: 512)'
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
            'default': 100,
            'metavar': 'N',
            'help': 'size of the hidden dimension (default: 100)'
        },
        {
            'name': '--output_dim',
            'type': int,
            'default': 1,
            'metavar': 'N',
            'help': 'size of the output dimension (default: 1)'
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
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMPredictor(args.input_dim, args.hidden_dim).to(device)

    print('Model loaded with input_dim {}, hidden_dim {}.'.format(
        args.input_dim, args.hidden_dim
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device, 10)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
