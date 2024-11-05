from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
import os
from torch import ones

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'raw')

def add_arbitrary_node_features(data):
    num_nodes = data.num_nodes
    data.x = ones((num_nodes, 1))
    return data

def load_clean_cora():
    cora_dataset = Planetoid(root=os.path.join(root_path, 'Planetoid'), name='Cora')
    return cora_dataset[0]

def load_clean_imdb():
    imdb_dataset = TUDataset(root=os.path.join(root_path, 'TUDataset'), name='IMDB-BINARY', pre_transform=add_arbitrary_node_features).shuffle()

    imdb_train_dataset = imdb_dataset[:750]
    imdb_test_dataset = imdb_dataset[750:]

    imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=96, shuffle=False)
    imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=96, shuffle=False)

    return imdb_dataset, imdb_train_loader, imdb_test_loader

def load_clean_enzymes():
    enzymes_dataset = TUDataset(root=os.path.join(root_path, 'TUDataset'), name='ENZYMES').shuffle()

    enzymes_train_dataset = enzymes_dataset[:450]
    enzymes_test_dataset = enzymes_dataset[450:]

    enzymes_train_loader = DataLoader(enzymes_train_dataset, batch_size=64, shuffle=False)
    enzymes_test_loader = DataLoader(enzymes_test_dataset, batch_size=64, shuffle=False)

    return enzymes_dataset, enzymes_train_loader, enzymes_test_loader
