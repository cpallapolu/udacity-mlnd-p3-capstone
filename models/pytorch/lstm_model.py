
import torch.nn as nn
import torch


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)

        out = self.fc(lstm_out)

        return self.relu(out.squeeze())

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        new_weight = weight.new(self.n_layers, batch_size, self.hidden_dim)

        return (new_weight.zero_().to(device), new_weight.zero_().to(device))
