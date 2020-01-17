
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dense(lstm_out)

        return self.sig(out.squeeze())
