from torch import ones, unique, tensor, long, randperm, zeros
import json

def load_model_config(filename):
    with open(filename, 'r') as file:
        model_config = json.load(file)
    return model_config

def generate_masks(num_nodes, train_ratio, val_ratio):
    # Generate a random permutation of node indices
    indices = randperm(num_nodes)

    # Compute the sizes of each split
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = num_nodes - train_size - val_size

    # Assign indices to each split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create masks
    train_mask = zeros(num_nodes, dtype=bool)
    val_mask = zeros(num_nodes, dtype=bool)
    test_mask = zeros(num_nodes, dtype=bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def add_arbitrary_node_features(data):
    num_nodes = data.num_nodes
    data.x = ones((num_nodes, 1))
    return data

def add_train_val_test_masks(data):
    train_mask, val_mask, test_mask = generate_masks(data.num_nodes, .10, .20)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def create_consecutive_mapping(data):
    unique_labels = unique(data.y)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    data.y = tensor([label_mapping[label.item()] for label in data.y], dtype=long)
    return data