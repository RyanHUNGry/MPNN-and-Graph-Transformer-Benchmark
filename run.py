from src.data import loader
from src.model.gcn import gcn
from src.model.gin import gin
from src.model.gat import gat
from src.model.gps import gps

def main():
    run_gcn_benchmarks()
    run_gin_benchmarks()
    run_gat_benchmarks()
    run_gps_benchmarks()

def run_gps_benchmarks():
    print("Running GPS Benchmarks")

    cora_data, num_classes = loader.load_clean_cora()
    gps_model = gps.GPS(cora_data, num_classes, 4, pe_channels=5, num_attention_heads=2, num_layers=3)
    gps.train(gps_model, cora_data)
    train_acc, test_acc = gps.test(gps_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gps_model = gps.GPS(pascalvoc_sp_data, num_classes, 4, pe_channels=5, num_attention_heads=2, num_layers=3)
    gps.train(gps_model, pascalvoc_sp_data)
    train_acc, test_acc = gps.test(gps_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gps_model = gps.GPS(enzymes_dataset, enzymes_dataset.num_classes, 4, pe_channels=5, num_attention_heads=2, num_layers=3)
    gps.train(gps_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gps.test(gps_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gps.test(gps_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gps_model = gps.GPS(imdb_dataset, imdb_dataset.num_classes, 4, pe_channels=5, num_attention_heads=2, num_layers=3)
    gps.train(gps_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gps.test(gps_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gps.test(gps_model, imdb_test_loader)}')

    print("-----------------------------------") 

def run_gat_benchmarks():
    print("Running GAT Benchmarks")

    cora_data, num_classes = loader.load_clean_cora()
    gat_model = gat.GAT(cora_data, num_classes)
    gat.train(gat_model, cora_data)
    train_acc, test_acc = gat.test(gat_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gat_model = gat.GAT(pascalvoc_sp_data, num_classes)
    gat.train(gat_model, pascalvoc_sp_data)
    train_acc, test_acc = gat.test(gat_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gat_model = gat.GAT(enzymes_dataset, enzymes_dataset.num_classes)
    gat.train(gat_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gat.test(gat_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gat.test(gat_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gat_model = gat.GAT(imdb_dataset, imdb_dataset.num_classes)
    gat.train(gat_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gat.test(gat_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gat.test(gat_model, imdb_test_loader)}')

    print("-----------------------------------")

def run_gin_benchmarks():
    print("Running GIN Benchmarks")

    cora_data, num_classes = loader.load_clean_cora()
    gin_model = gin.GIN(cora_data, num_classes, hidden_layers=3)
    gin.train(gin_model, cora_data)
    train_acc, test_acc = gin.test(gin_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gin_model = gin.GIN(pascalvoc_sp_data, num_classes, hidden_layers=3)
    gin.train(gin_model, pascalvoc_sp_data)
    train_acc, test_acc = gin.test(gin_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')

    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gin_model = gin.GIN(enzymes_dataset, enzymes_dataset.num_classes, hidden_channels=32, hidden_layers=3)
    gin.train(gin_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gin.test(gin_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gin.test(gin_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gin_model = gin.GIN(imdb_dataset, imdb_dataset.num_classes, hidden_channels=32, hidden_layers=3)
    gin.train(gin_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gin.test(gin_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gin.test(gin_model, imdb_test_loader)}')

    print("-----------------------------------")

def run_gcn_benchmarks():
    print("Running GCN Benchmarks")

    cora_data, num_classes = loader.load_clean_cora()
    gcn_model = gcn.GCN(cora_data, num_classes)
    gcn.train(gcn_model, cora_data)
    train_acc, test_acc = gcn.test(gcn_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    pascalvoc_sp_data, num_classes = loader.load_pascalvoc_sp()
    gcn_model = gcn.GCN(pascalvoc_sp_data, num_classes)
    gcn.train(gcn_model, pascalvoc_sp_data)
    train_acc, test_acc = gcn.test(gcn_model, pascalvoc_sp_data)
    print(f'Train Accuracy for PascalVOC-SP node-level classification: {train_acc}')
    print(f'Test Accuracy for PascalVOC-SP node-level classification: {test_acc}')
    
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gcn_model = gcn.GCN(enzymes_dataset, enzymes_dataset.num_classes)
    gcn.train(gcn_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gcn.test(gcn_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gcn.test(gcn_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gcn_model = gcn.GCN(imdb_dataset, imdb_dataset.num_classes)
    gcn.train(gcn_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gcn.test(gcn_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gcn.test(gcn_model, imdb_test_loader)}')

    print("-----------------------------------")

if __name__ == '__main__':
    main()
