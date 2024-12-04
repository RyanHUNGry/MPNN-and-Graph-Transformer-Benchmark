import torch
from torch.nn import ModuleList, Linear, LogSoftmax
from torch_geometric.nn import GCNConv, GPSConv, global_add_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import Data

import torch.nn as nn

"""
GPS model from the paper "Recipe for a General, Powerful, Scalable Graph Transformer"
by Ladislav Rampášek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini (https://arxiv.org/abs/2205.12454)

The GPS class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a summation readout for each node embedding at each layer, and then concatenate the readouts.

Tunable model hyperparameters include the number of hidden channels, positional encoding channels, attention heads, and hidden layers.
"""
class GPS(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=4, pe_channels=4, num_attention_heads=1, num_layers=2):
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

        if type(data) is not Data:
            self.readout = global_add_pool
            return

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # concatenate positional encodings
        pe = torch.empty(x.shape[0], 0)
        if hasattr(data, "random_walk_pe"):
            pe = torch.cat([pe, data.random_walk_pe], dim=1)
        if hasattr(data, "laplacian_eigenvector_pe"):
            pe = torch.cat([pe, data.laplacian_eigenvector_pe], dim=1)

        pe = self.pe_lin(pe)
        pe = self.pe_norm(pe)
        x = self.input_lin(x)
        x = torch.cat((x, pe), dim=1)
        for layer in self.layers:
            x = layer(x, edge_index)
        
        if type(data) is not Data:
            x = self.readout(x, data.batch)

        x = self.lin(x)
        x = self.output(x)
        return x
    
def train(gps, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gps.parameters(), lr=0.01, weight_decay=5e-4)
    gps.train()
    for epoch in range(100):
        if type(data) is not Data:
            for batch in data:
                out = gps(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            out = gps(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(gps, data):
    gps.eval()
    if type(data) is not Data:
        correct = 0
        for batch in data:
            out = gps(batch)  
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
        return correct / len(data.dataset)
    pred = gps(data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / data.train_mask.sum()
    if isinstance(train_acc, torch.Tensor):
        train_acc = train_acc.item()
    if isinstance(test_acc, torch.Tensor):
        test_acc = test_acc.item()
    return train_acc, test_acc
