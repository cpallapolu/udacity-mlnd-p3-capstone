
import argparse
import json
import os
import pickle
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier


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
    model = LSTMClassifier(
        model_info['embedding_dim'],
        model_info['hidden_dim'],
        model_info['vocab_size']
        )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print('Done loading model.')
    return model


def _get_train_data_loader(batch_size, training_dir):
    print('Get train data loader.')

    train_data = pd.read_csv(
        os.path.join(training_dir, 'train.csv'),
        header=None,
        names=None
    )

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
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
        print('Epoch: {}, BCELoss: {}'.format(
            epoch, total_loss / len(train_loader)
        ))


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
            'name': '--embedding_dim',
            'type': int,
            'default': 32,
            'metavar': 'N',
            'help': 'size of the word embeddings (default: 32)'
        },
        {
            'name': '--hidden_dim',
            'type': int,
            'default': 100,
            'metavar': 'N',
            'help': 'size of the hidden dimension (default: 100)'
        },
        {
            'name': '--vocab_size',
            'type': int,
            'default': 5000,
            'metavar': 'N',
            'help': 'size of the vocabulary (default: 5000)'
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
            parser_arg.name,
            type=parser_arg.type,
            default=parser_arg.default,
            metavar=parser_arg.metavar,
            help=parser_arg.help
        )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}.'.format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMClassifier(
        args.embedding_dim,
        args.hidden_dim,
        args.vocab_size
    ).to(device)

    print('Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.'
          .format(args.embedding_dim, args.hidden_dim, args.vocab_size))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

    # Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
