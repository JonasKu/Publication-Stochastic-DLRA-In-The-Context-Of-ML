from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.low_rank_neural_networks.train import Trainer
import argparse
import toml
import numpy as np
import matplotlib.pyplot as plt

def parseInput():
    '''Parse input arguments to determine config name'''
    parser = argparse.ArgumentParser(description='Low-rank trainer for neural networks. Specify config file or provide option for automatic config file search.')
    parser.add_argument('-c', '--config_file', default='bm_net_60.toml', help='Specify name of config file. Default is config.toml.')
    parser.add_argument('--find_all', type=bool, default=True, help='Set flag if config files should be identified by program.')
    parser.add_argument('-f', '--folder', default='runs_compare_BM', help='Specify name of folder for config file search.')
    args = parser.parse_args()
    flag = args.find_all
    if flag:
        path = Path(args.folder)
        return list(path.rglob("*.toml"))
    else:
        return [Path(args.config_file)]


def read_configuration(filePath):
    try:
        # Read the TOML file
        with open(filePath, 'r') as file:
            config = toml.load(file)

        # Extract learning rate and dense architecture from the configuration
        settings = config.get('settings', {})
        learningRate = settings.get('learningRate')
        batchSize = settings.get('batchSize')
        numEpochs = settings.get('numEpochs')
        outputDir = settings.get('outputDir','architecture')
        logName = settings.get('logName','current')

        architecture = config.get('layer')

        if learningRate is None or architecture is None:
            raise ValueError("The TOML file is missing required fields.")

        return learningRate, batchSize, numEpochs, outputDir, logName, architecture

    except FileNotFoundError:
        print(f"Error: File not found - {filePath}")
        exit()
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    tomlFiles = parseInput()
    print(tomlFiles)
    for tomlFile in tomlFiles:

        learningRate, batchSize, numEpochs, outputDir, logName, architecture = read_configuration(tomlFile)

        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        trainDataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

        testDataset = datasets.MNIST(root='./data', train=False, transform=transform)
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

        # create model directory from config location

        # check if model directory will match any of the existing directories 
        # if yes, rename directory
        systemDirs = ["data", "src", "tests"]
        nameDir = Path(tomlFile).stem
        if nameDir in systemDirs:
            newDir = nameDir + "_model"
            print(f"Renaming model directory from {nameDir} to {newDir}")
            nameDir = newDir

        modelDir = Path(tomlFile).resolve().parent / nameDir # modelDir is config name without .toml ending
        if not modelDir.is_dir():
            modelDir.mkdir()

        # train neural network
        trainer = Trainer(architecture, modelDir, logName, tomlFile, trainLoader, testLoader)
        layer_ranks, test_accs = trainer.train(numEpochs, learningRate)

        test_epochs = np.linspace(1, numEpochs, len(test_accs))

        # Save the fine-tuning results
        np.savez(f"training_results_{tomlFile.name}.npz", test_epochs=test_epochs, test_accs=test_accs)

        print(f"-> Training for {tomlFile} finished. Plot saved as 'accuracy_bm.png'.")

    
    data = np.load("training_results_bm_net_60_1.toml.npz", allow_pickle=True)
    test_epochs_bm_1 = data["test_epochs"]
    test_accs_bm_1 = data["test_accs"]
    data = np.load("training_results_bm_net_60_2.toml.npz", allow_pickle=True)
    test_epochs_bm_2 = data["test_epochs"]
    test_accs_bm_2 = data["test_accs"]
    data = np.load("training_results_bm_net_60_3.toml.npz", allow_pickle=True)
    test_epochs_bm_3 = data["test_epochs"]
    test_accs_bm_3 = data["test_accs"]
    data = np.load("training_results_bm_net_60_4.toml.npz", allow_pickle=True)
    test_epochs_bm_4 = data["test_epochs"]
    test_accs_bm_4 = data["test_accs"]
    data = np.load("training_results_dlrt_net_60.toml.npz", allow_pickle=True)
    test_epochs_dlrt = data["test_epochs"]
    test_accs_dlrt = data["test_accs"]

    fig, ax = plt.subplots()

    # Plot test accuracy
    ax.plot(test_epochs_dlrt, test_accs_dlrt, color='black', linestyle='-', marker='o',
        label="DLRT, h = 1e-1", linewidth=2)

    ax.plot(test_epochs_bm_1, test_accs_bm_1, color='tab:red', linestyle='--', marker='s',
        label="BM, h = 1e-1", linewidth=2)

    ax.plot(test_epochs_bm_2, test_accs_bm_2, color='tab:blue', linestyle='-.', marker='^',
        label="BM, h = 1e-3", linewidth=2)

    ax.plot(test_epochs_bm_3, test_accs_bm_3, color='tab:green', linestyle=':', marker='D',
        label="BM, h = 1e-4", linewidth=2)

    ax.plot(test_epochs_bm_4, test_accs_bm_4, color='tab:purple', linestyle='-', marker='x',
        label="BM, h = 1e-5", linewidth=2)
    
    # Labeling
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.tick_params(axis='y')
    ax.grid(True)
    ax.legend(loc='lower right')

    fig.tight_layout()

    # Save and show the plot
    plt.savefig("accuracy_bm_vs_dlrt.png", dpi=300)
    plt.show()

    
