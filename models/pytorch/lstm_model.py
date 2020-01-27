
import torch.nn as nn
import torch

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        out = self.dropout(lstm_out)
        
        out = self.fc(lstm_out)

        return self.sig(out.squeeze())
