import torch
from torch_geometric.data.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""
GCN model from the paper "Semi-Supervised Classification with Graph Convolutional Networks" 
by Thomas N. Kipf and Max Welling (https://arxiv.org/pdf/1609.02907)

The GCN class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a global mean pooling to the node embeddings and then apply a linear layer to the pooled embeddings.

Tunable model hyperparameters include the number of hidden channels.
"""
class GCN(torch.nn.Module):
    def __init__(self, data, hidden_channels=16, hidden_layers=2):
        super().__init__()
        if not type(data) is Data:
            self.conv1 = GCNConv(data.num_node_features, hidden_channels)
            self.convs = torch.nn.ModuleList()
            for i in range(hidden_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.lin = torch.nn.Linear(hidden_channels, data.num_classes)
        else:
            num_classes = len(data.y.unique())
            num_node_features = data.num_node_features
            
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.convs = torch.nn.ModuleList()
            for i in range(hidden_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels if i != hidden_layers - 2 else num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        if not type(data) is Data:
            x = F.relu(x)
            x = global_mean_pool(x, data.batch)
            x = F.dropout(x, training=self.training)
            x = self.lin(x)
            return x

        return F.log_softmax(x, dim=1)

def train(gcn, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    gcn.train()
    for epoch in range(200):
        if not type(data) is Data:
            for batch in data:
                out = gcn(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            out = gcn(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(gcn, data):
    gcn.eval()
    if not type(data) is Data:
        correct = 0
        for batch in data:
            out = gcn(batch)  
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
        return correct / len(data.dataset)
    
    pred = gcn(data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / data.train_mask.sum()
    return train_acc, test_acc
