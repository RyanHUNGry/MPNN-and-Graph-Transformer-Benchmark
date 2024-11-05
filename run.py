from src.data import loader
from src.model.gcn import gcn
from src.model.gin import gin
from src.model.gat import gat

def main():
    run_gcn_benchmarks()
    run_gin_benchmarks()
    run_gat_benchmarks()

def run_gat_benchmarks():
    print("Running GAT Benchmarks")
    cora_data = loader.load_clean_cora()
    gat_model = gat.GAT(cora_data)
    gat.train(gat_model, cora_data)
    train_acc, test_acc = gat.test(gat_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gat_model = gat.GAT(enzymes_dataset)
    gat.train(gat_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gat.test(gat_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gat.test(gat_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gat_model = gat.GAT(imdb_dataset)
    gat.train(gat_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gat.test(gat_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gat.test(gat_model, imdb_test_loader)}')

    print("-----------------------------------")

def run_gin_benchmarks():
    print("Running GIN Benchmarks")

    cora_data = loader.load_clean_cora()
    gin_model = gin.GIN(cora_data, hidden_layers=3)
    gin.train(gin_model, cora_data)
    train_acc, test_acc = gcn.test(gin_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')

    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gin_model = gin.GIN(enzymes_dataset, hidden_channels=32, hidden_layers=3)
    gin.train(gin_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gin.test(gin_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gin.test(gin_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gin_model = gin.GIN(imdb_dataset, hidden_channels=32, hidden_layers=3)
    gin.train(gin_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gin.test(gin_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gin.test(gin_model, imdb_test_loader)}')

    print("-----------------------------------")

def run_gcn_benchmarks():
    print("Running GCN Benchmarks")

    cora_data = loader.load_clean_cora()
    gcn_model = gcn.GCN(cora_data)
    gcn.train(gcn_model, cora_data)
    train_acc, test_acc = gcn.test(gcn_model, cora_data)
    print(f'Train Accuracy for Cora node-level classification: {train_acc}')
    print(f'Test Accuracy for Cora node-level classification: {test_acc}')
    
    enzymes_dataset, enzymes_train_loader, enzymes_test_loader = loader.load_clean_enzymes()
    gcn_model = gcn.GCN(enzymes_dataset)
    gcn.train(gcn_model, enzymes_train_loader)
    print(f'Train Accuracy for Enzymes graph-level classification: {gcn.test(gcn_model, enzymes_train_loader)}')
    print(f'Test Accuracy for Enzymes graph-level classification: {gcn.test(gcn_model, enzymes_test_loader)}')

    imdb_dataset, imdb_train_loader, imdb_test_loader = loader.load_clean_imdb()
    gcn_model = gcn.GCN(imdb_dataset)
    gcn.train(gcn_model, imdb_train_loader)
    print(f'Train Accuracy for IMDB graph-level classification: {gcn.test(gcn_model, imdb_train_loader)}')
    print(f'Test Accuracy for IMDB graph-level classification: {gcn.test(gcn_model, imdb_test_loader)}')

    print("-----------------------------------")

if __name__ == '__main__':
    main()