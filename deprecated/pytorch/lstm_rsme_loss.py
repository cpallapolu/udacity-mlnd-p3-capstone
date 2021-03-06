
import torch.nn as nn
import torch


class RSMELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()

        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y))
        return loss
