import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    Design the AutoEncoder architecure

    Architecture:
    Encoder: input_dim -> hidden_layer_1 -> hidden_layer_2 -> bottleneck_size
    Decoder: bottleneck_size -> hidden_layer_3 -> hidden_layer_4 -> input_dim
    Hidden layer 1 <-> hidden layer 3, hidden layer 2 <-> hidden layer 4
    """
    def __init__(self, input_dim, h1, h2, b):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, b), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(b, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, self.input_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded