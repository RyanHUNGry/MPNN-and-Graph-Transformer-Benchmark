import torch
from torch_geometric.data.data import Data
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool

"""
GIN model from the paper "How Powerful are Graph Neural Networks?"
by Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka (https://arxiv.org/pdf/1810.00826)

The GIN class and train/test methods, by default, performs node-level classification, but it will use the data instance
type to determine whether to perform graph-level classification if data is not an instance of torch_geometric.data.data.Data.
Graph-level classification will apply a summation readout for each node embedding at each layer, and then concatenate the readouts.

Tunable model hyperparameters include the number of hidden channels and hidden layers.
"""
class GIN(torch.nn.Module):
    def __init__(self, data, num_classes, hidden_channels=16, hidden_layers=5):
        super().__init__()

        num_node_features = data.num_node_features

        self.conv1 = GINConv(
            Sequential(
                Linear(num_node_features, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels)
            )
        )

        self.convs = torch.nn.ModuleList()
        for i in range(hidden_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BatchNorm1d(hidden_channels),
                    ),
                )
            )

        if type(data) is Data:
            self.lin = Linear(hidden_channels, num_classes)
            return
        else:
            self.lin1 = Linear(hidden_channels*hidden_layers, hidden_channels*hidden_layers)
            self.lin2 = Linear(hidden_channels*hidden_layers, num_classes)
            return

    def forward(self, data):
        if type(data) is Data:
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            
            for conv in self.convs:
                x = conv(x, edge_index)
            
            return F.log_softmax(x, dim=1)
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

            readouts = []

            x = self.conv1(x, edge_index)
            readouts.append(global_add_pool(x, batch))

            for conv in self.convs:
                x = conv(x, edge_index)
                readouts.append(global_add_pool(x, batch))

            concatenated_readouts = torch.cat(tuple(readouts), dim=1)

            concatenated_readouts = self.lin1(concatenated_readouts)
            concatenated_readouts = concatenated_readouts.relu()
            concatenated_readouts = F.dropout(concatenated_readouts, training=self.training)
            concatenated_readouts = self.lin2(concatenated_readouts)
        
            return F.log_softmax(concatenated_readouts, dim=1)

def train(gin, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gin.parameters(), lr=0.01, weight_decay=5e-4)
    gin.train()
    for epoch in range(100):
        if not type(data) is Data:
            for batch in data:
                out = gin(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            out = gin(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(gin, data):
    gin.eval()
    if not type(data) is Data:
        correct = 0
        for batch in data:
            out = gin(batch)  
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
        return correct / len(data.dataset)
    
    pred = gin(data).argmax(dim=1)
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / data.train_mask.sum()
    if isinstance(train_acc, torch.Tensor):
        train_acc = train_acc.item()
    if isinstance(test_acc, torch.Tensor):
        test_acc = test_acc.item()
    return train_acc, test_acc
