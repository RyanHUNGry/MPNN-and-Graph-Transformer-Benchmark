from torch_geometric.datasets import Planetoid, TUDataset, LRGBDataset
from torch_geometric.loader import DataLoader
import os
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE
from src.util.util import add_arbitrary_node_features, add_train_val_test_masks, create_consecutive_mapping

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'raw')

def load_clean_cora(transformations=None):
    cora_dataset = Planetoid(root=os.path.join(root_path, 'Planetoid'), name='Cora', pre_transform=Compose([AddLaplacianEigenvectorPE(2)]))
    return cora_dataset[0], cora_dataset.num_classes

"""
Data does not have features, so add arbitrary scalar one for models to work.
"""
def load_clean_imdb():
    imdb_dataset = TUDataset(root=os.path.join(root_path, 'TUDataset'), name='IMDB-BINARY', pre_transform=Compose([add_arbitrary_node_features, AddLaplacianEigenvectorPE(2)])).shuffle()

    imdb_train_dataset = imdb_dataset[:750]
    imdb_test_dataset = imdb_dataset[750:]

    imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=96, shuffle=False)
    imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=96, shuffle=False)

    return imdb_dataset, imdb_train_loader, imdb_test_loader

def load_clean_enzymes():
    enzymes_dataset = TUDataset(root=os.path.join(root_path, 'TUDataset'), name='ENZYMES', pre_transform=Compose([AddLaplacianEigenvectorPE(2)])).shuffle()

    enzymes_train_dataset = enzymes_dataset[:450]
    enzymes_test_dataset = enzymes_dataset[450:]

    enzymes_train_loader = DataLoader(enzymes_train_dataset, batch_size=64, shuffle=False)
    enzymes_test_loader = DataLoader(enzymes_test_dataset, batch_size=64, shuffle=False)

    return enzymes_dataset, enzymes_train_loader, enzymes_test_loader

"""
Data has 21 classes, but many individual records only have a few classes. So, add consecutive mapping to ensure cross-entropy loss works, and add train/val/test masks.
"""
def load_pascalvoc_sp():
    pascalvoc_sp_dataset = LRGBDataset(root=os.path.join(root_path, 'LRGBDataset'), name='PascalVOC-SP', pre_transform=Compose([add_train_val_test_masks, create_consecutive_mapping, AddLaplacianEigenvectorPE(2)]))
    return pascalvoc_sp_dataset[3], pascalvoc_sp_dataset.num_classes # random 3rd indexed dataset, but can configure
