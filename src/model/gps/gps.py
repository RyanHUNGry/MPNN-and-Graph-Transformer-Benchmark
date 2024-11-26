import torch
from torch.nn import ModuleList, Linear, LogSoftmax
from torch_geometric.nn import GCNConv, GPSConv
from torch_geometric.datasets import Planetoid

import torch.nn as nn

"""
GPS model from the paper "Recipe for a General, Powerful, Scalable Graph Transformer"
by Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini (https://arxiv.org/abs/2205.12454)

The GPS class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a summation readout for each node embedding at each layer, and then concatenate the readouts.

Tunable model hyperparameters include the number of hidden channels, hidden layers, positional encoding channels, and number of attention heads.
"""
class GPS(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels, pe_channels=4, num_attention_heads=1, num_layers=2):
        super().__init__()
        
        self.pe_lin = nn.Linear(pe_channels, hidden_channels)
        self.pe_norm = nn.LayerNorm(hidden_channels)
        self.input_lin = nn.Linear(data.num_node_features, hidden_channels)

        self.layers = ModuleList()
        hidden_channels *= 2
        for _ in range(num_layers):
            mpnn = GCNConv(hidden_channels, hidden_channels)
            transformer = GPSConv(hidden_channels, mpnn, heads=num_attention_heads)
            self.layers.append(transformer)

        self.lin = Linear(hidden_channels, num_classes)
        self.output = LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index, laplacian_eigenvector_pe = data.x, data.edge_index, data.laplacian_eigenvector_pe
        pe = self.pe_lin(laplacian_eigenvector_pe)
        pe = self.pe_norm(pe)
        x = self.input_lin(x)
        x = torch.cat((x, pe), dim=1)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.lin(x)
        x = self.output(x)
        return x
    
def train(gps, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gps.parameters(), lr=0.01, weight_decay=5e-4)
    gps.train()
    for epoch in range(100):
        print(epoch)
        out = gps(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
def test(gps, data):
    gps.eval()
    pred = gps(data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / data.train_mask.sum()
    return train_acc, test_acc
