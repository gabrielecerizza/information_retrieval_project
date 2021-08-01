import numpy as np
import torch
from torch import nn
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

class AutoEncoder(nn.Module):
    def __init__(
        self, 
        dims: List[int]
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(True),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(True),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(True),
            nn.Linear(dims[3], dims[4])
        )
        self.decoder = nn.Sequential(
            nn.Linear(dims[4], dims[3]),
            nn.ReLU(True),
            nn.Linear(dims[3], dims[2]),
            nn.ReLU(True),
            nn.Linear(dims[2], dims[1]),
            nn.ReLU(True),
            nn.Linear(dims[1], dims[0])
        )

    def forward(self, X: np.ndarray):
        X = self.encoder(X)
        X = self.decoder(X)
        return X