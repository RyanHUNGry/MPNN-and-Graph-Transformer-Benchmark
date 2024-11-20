import torch
from torch_geometric.data.data import Data
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ELU, BatchNorm1d
from torch_geometric.nn import GATConv, global_add_pool

"""
GAT model from the paper "Graph Attention Networks" 
by Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (https://arxiv.org/pdf/1710.10903)

The GAT class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a summation readout on final node embeddings and then apply a linear layer to the pooled embeddings.

Tunable model hyperparameters include the number of hidden channels and hidden layers.
"""
class GAT(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=8, heads=8, hidden_layers=2):
        super().__init__()

        if type(data) is Data:
            num_node_features = data.num_node_features

            self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
            self.convs = torch.nn.ModuleList()
            for i in range(hidden_layers - 1):
                self.convs.append(GATConv(hidden_channels*heads, hidden_channels if i != hidden_layers - 2 else num_classes, heads=heads if i != hidden_layers - 2 else 1))
        else:
            self.conv1 = GATConv(data.num_node_features, hidden_channels, heads=heads)
            self.convs = torch.nn.ModuleList()
            for i in range(hidden_layers - 1):
                self.convs.append(GATConv(hidden_channels*heads, hidden_channels if i != hidden_layers - 2 else num_classes, heads=heads if i != hidden_layers - 2 else 1))
            self.lin = Linear(num_classes, num_classes)

    def forward(self, data):
        if type(data) is Data:
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            for conv in self.convs:
                F.elu(x)
                x = conv(x, edge_index)
            
            return F.log_softmax(x, dim=1)
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            for conv in self.convs:
                F.elu(x)
                x = conv(x, edge_index)

            x = global_add_pool(x, batch)
            x = F.dropout(x, training=self.training)
            x = self.lin(x)
            return x

def train(gat, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=5e-4)
    gat.train()
    for _ in range(200):
        if not type(data) is Data:
            for batch in data:
                out = gat(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            out = gat(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(gat, data):
    gat.eval()
    if not type(data) is Data:
        correct = 0
        for batch in data:
            out = gat(batch)  
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
        return correct / len(data.dataset)
    
    pred = gat(data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / data.train_mask.sum()
    return train_acc, test_acc
