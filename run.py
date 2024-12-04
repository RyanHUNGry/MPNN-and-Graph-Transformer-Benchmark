from src.data import loader
from src.model.gcn import gcn
from src.model.gin import gin
from src.model.gat import gat
from src.model.gps import gps
from src.util.util import load_model_config
import os
import json

def main():
    parameters_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'params.json')
    parameters = load_model_config(parameters_path)

    res = {}
    res["GAT_VERSUS_GCN"] = run_gat_versus_gcn_benchmarks() # Model performance with respect to # of hidden layers
    res["GCN"] = run_gcn_benchmarks(parameters['GCN'])
    res["GIN"] = run_gin_benchmarks(parameters['GIN'])
    res["GAT"] = run_gat_benchmarks(parameters['GAT'])
    res["GPS"] = run_gps_benchmarks(parameters['GPS'])

    res_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs', 'results.json')

    with open(res_path, 'w') as f:
        json.dump(res, f, indent=4)

def run_gat_versus_gcn_benchmarks():
    print("Running GAT vs GCN Benchmarks")

    cora_data, num_classes = loader.load_clean_cora()
    hidden_layers = [2, 4, 6, 8]
    gcn_train_test_accs, gat_train_test_accs = [None] * 4, [None] * 4

    res = {}

    for i, hidden_layer in enumerate(hidden_layers):
        gat_model = gat.GAT(cora_data, num_classes, hidden_layers=hidden_layer)
        gat.train(gat_model, cora_data)
        train_acc, test_acc = gat.test(gat_model, cora_data)
        gat_train_test_accs[i] = (train_acc, test_acc)

        gcn_model = gcn.GCN(cora_data, num_classes, hidden_layers=hidden_layer)
        gcn.train(gcn_model, cora_data)
        train_acc, test_acc = gcn.test(gcn_model, cora_data)
        gcn_train_test_accs[i] = (train_acc, test_acc)

    for i, hidden_layer in enumerate(hidden_layers):
        print(f'Hidden Layers: {hidden_layer}')
        print(f'GAT Train Accuracy: {gat_train_test_accs[i][0]}')
        print(f'GAT Test Accuracy: {gat_train_test_accs[i][1]}')
        print(f'GCN Train Accuracy: {gcn_train_test_accs[i][0]}')
        print(f'GCN Test Accuracy: {gcn_train_test_accs[i][1]}')
        print("-----------------------------------")

    res = {
        "gat_train_test_accs": gat_train_test_accs,
        "gcn_train_test_accs": gcn_train_test_accs,
        "hidden_layers": hidden_layers
    }

    print("-----------------------------------")
    return res


def run_gps_benchmarks(parameters):
    print("Running GPS Benchmarks")

    res = {}

    cora_params = parameters['Cora']
    cora_data, num_classes = loader.load_clean_cora()
    gps_model = gps.GPS(cora_data, num_classes, hidden_channels=cora_params['hidden_channels'], pe_channels=cora_params['pe_channels'], num_attention_heads=cora_params['num_attention_heads'], num_layers=cora_params['num_layers'])
    gps.train(gps_model, cora_data)
    train_acc, test_acc = gps.test(gps_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    res["Cora"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": cora_params
    }

    pascalvoc_sp_params = parameters['PascalVOC-SP']
    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gps_model = gps.GPS(pascalvoc_sp_data, num_classes, hidden_channels=pascalvoc_sp_params['hidden_channels'], pe_channels=pascalvoc_sp_params['pe_channels'], num_attention_heads=pascalvoc_sp_params['num_attention_heads'], num_layers=pascalvoc_sp_params['num_layers'])
    gps.train(gps_model, pascalvoc_sp_data)
    train_acc, test_acc = gps.test(gps_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    res["PascalVOC-SP"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": pascalvoc_sp_params
    }

    enzymes_params = parameters['Enzymes']
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gps_model = gps.GPS(enzymes_dataset, enzymes_dataset.num_classes, hidden_channels=enzymes_params['hidden_channels'], pe_channels=enzymes_params['pe_channels'], num_attention_heads=enzymes_params['num_attention_heads'], num_layers=enzymes_params['num_layers'])
    gps.train(gps_model, enzymes_train_loader)
    train_acc, test_acc = gps.test(gps_model, enzymes_train_loader), gps.test(gps_model, enzymes_test_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {train_acc}')
    print(f'Test Accuracy for Enzymes graph-level classification: {test_acc}')

    res["Enzymes"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": enzymes_params
    }

    imdb_params = parameters['IMDB-BINARY']
    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gps_model = gps.GPS(imdb_dataset, imdb_dataset.num_classes, hidden_channels=imdb_params['hidden_channels'], pe_channels=imdb_params['pe_channels'], num_attention_heads=imdb_params['num_attention_heads'], num_layers=imdb_params['num_layers'])
    gps.train(gps_model, imdb_train_loader)
    train_acc, test_acc = gps.test(gps_model, imdb_train_loader), gps.test(gps_model, imdb_test_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {train_acc}')
    print(f'Test Accuracy for IMDB graph-level classification: {test_acc}')

    res["IMDB-BINARY"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": imdb_params
    }

    print("-----------------------------------")

    return res

def run_gat_benchmarks(parameters):
    print("Running GAT Benchmarks")

    res = {}

    cora_params = parameters['Cora']
    cora_data, num_classes = loader.load_clean_cora()
    gat_model = gat.GAT(cora_data, num_classes, hidden_channels=cora_params['hidden_channels'], hidden_layers=cora_params['hidden_layers'], heads=cora_params['heads'])
    gat.train(gat_model, cora_data)
    train_acc, test_acc = gat.test(gat_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    res["Cora"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": cora_params
    }

    pascalvoc_sp_params = parameters['PascalVOC-SP']
    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gat_model = gat.GAT(pascalvoc_sp_data, num_classes, hidden_channels=pascalvoc_sp_params['hidden_channels'], hidden_layers=pascalvoc_sp_params['hidden_layers'], heads=pascalvoc_sp_params['heads'])
    gat.train(gat_model, pascalvoc_sp_data)
    train_acc, test_acc = gat.test(gat_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    res["PascalVOC-SP"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": pascalvoc_sp_params
    }

    enzymes_params = parameters['Enzymes']
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gat_model = gat.GAT(enzymes_dataset, enzymes_dataset.num_classes, hidden_channels=enzymes_params['hidden_channels'], hidden_layers=enzymes_params['hidden_layers'], heads=enzymes_params['heads'])
    gat.train(gat_model, enzymes_train_loader)
    train_acc, test_acc = gat.test(gat_model, enzymes_train_loader), gat.test(gat_model, enzymes_test_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {train_acc}')
    print(f'Test Accuracy for Enzymes graph-level classification: {test_acc}')

    res["Enzymes"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": enzymes_params
    }

    imdb_params = parameters['IMDB-BINARY']
    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gat_model = gat.GAT(imdb_dataset, imdb_dataset.num_classes, hidden_channels=imdb_params['hidden_channels'], hidden_layers=imdb_params['hidden_layers'], heads=imdb_params['heads'])
    gat.train(gat_model, imdb_train_loader)
    train_acc, test_acc = gat.test(gat_model, imdb_train_loader), gat.test(gat_model, imdb_test_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {train_acc}')
    print(f'Test Accuracy for IMDB graph-level classification: {test_acc}')

    res["IMDB-BINARY"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": imdb_params
    }

    print("-----------------------------------")
    return res

def run_gin_benchmarks(parameters):
    print("Running GIN Benchmarks")

    res = {}

    cora_params = parameters['Cora']
    cora_data, num_classes = loader.load_clean_cora()
    gin_model = gin.GIN(cora_data, num_classes, hidden_channels=cora_params['hidden_channels'], hidden_layers=cora_params['hidden_layers'])
    gin.train(gin_model, cora_data)
    train_acc, test_acc = gin.test(gin_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    res["Cora"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": cora_params
    }

    pascalvoc_params = parameters['PascalVOC-SP']
    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gin_model = gin.GIN(pascalvoc_sp_data, num_classes, hidden_channels=pascalvoc_params['hidden_channels'], hidden_layers=pascalvoc_params['hidden_layers'])
    gin.train(gin_model, pascalvoc_sp_data)
    train_acc, test_acc = gin.test(gin_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    res["PascalVOC-SP"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": pascalvoc_params
    }

    enzymes_params = parameters['Enzymes']
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gin_model = gin.GIN(enzymes_dataset, enzymes_dataset.num_classes, hidden_channels=enzymes_params['hidden_channels'], hidden_layers=enzymes_params['hidden_layers'])
    gin.train(gin_model, enzymes_train_loader)
    train_acc, test_acc = gin.test(gin_model, enzymes_train_loader), gin.test(gin_model, enzymes_test_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {train_acc}')
    print(f'Test Accuracy for Enzymes graph-level classification: {test_acc}')

    res["Enzymes"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": enzymes_params
    }

    imdb_params = parameters['IMDB-BINARY']
    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gin_model = gin.GIN(imdb_dataset, imdb_dataset.num_classes, hidden_channels=imdb_params['hidden_channels'], hidden_layers=imdb_params['hidden_layers'])
    gin.train(gin_model, imdb_train_loader)
    train_acc, test_acc = gin.test(gin_model, imdb_train_loader), gin.test(gin_model, imdb_test_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {train_acc}')
    print(f'Test Accuracy for IMDB graph-level classification: {test_acc}')

    res["IMDB-BINARY"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": imdb_params
    }

    print("-----------------------------------")
    return res

def run_gcn_benchmarks(parameters):
    print("Running GCN Benchmarks")

    res = {}

    cora_params = parameters['Cora']
    cora_data, num_classes = loader.load_clean_cora()
    gcn_model = gcn.GCN(cora_data, num_classes, hidden_channels=cora_params['hidden_channels'], hidden_layers=cora_params['hidden_layers'])
    gcn.train(gcn_model, cora_data)
    train_acc, test_acc = gcn.test(gcn_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    res["Cora"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": cora_params
    }

    pascalvoc_params = parameters['PascalVOC-SP']
    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gcn_model = gcn.GCN(pascalvoc_sp_data, num_classes, hidden_channels=pascalvoc_params['hidden_channels'], hidden_layers=pascalvoc_params['hidden_layers'])
    gcn.train(gcn_model, pascalvoc_sp_data)
    train_acc, test_acc = gcn.test(gcn_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    res["PascalVOC-SP"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": pascalvoc_params
    }

    enzymes_params = parameters['Enzymes']
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gcn_model = gcn.GCN(enzymes_dataset, enzymes_dataset.num_classes, hidden_channels=enzymes_params['hidden_channels'], hidden_layers=enzymes_params['hidden_layers'])
    gcn.train(gcn_model, enzymes_train_loader)
    train_acc, test_acc = gcn.test(gcn_model, enzymes_train_loader), gcn.test(gcn_model, enzymes_test_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {train_acc}')
    print(f'Test Accuracy for Enzymes graph-level classification: {test_acc}')

    res["Enzymes"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": enzymes_params
    }

    imdb_params = parameters['IMDB-BINARY']
    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gcn_model = gcn.GCN(imdb_dataset, imdb_dataset.num_classes, hidden_channels=imdb_params['hidden_channels'], hidden_layers=imdb_params['hidden_layers'])
    gcn.train(gcn_model, imdb_train_loader)
    train_acc, test_acc = gcn.test(gcn_model, imdb_train_loader), gcn.test(gcn_model, imdb_test_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {train_acc}')
    print(f'Test Accuracy for IMDB graph-level classification: {test_acc}')

    res["IMDB-BINARY"] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "params": imdb_params
    }

    print("-----------------------------------")
    return res

if __name__ == '__main__':
    main()
