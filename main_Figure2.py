from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.low_rank_neural_networks.train import Trainer
import argparse
import toml
import numpy as np
import matplotlib.pyplot as plt


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
    tomlFiles = [Path("runs_compare_rank/run_1.toml"),Path("runs_compare_rank/run_2.toml"),Path("runs_compare_rank/run_3.toml"),Path("runs_compare_rank/run_4.toml"),Path("runs_compare_rank/run_5.toml"),Path("runs_compare_rank/run_6.toml"),Path("runs_compare_rank/run_7.toml"),Path("runs_compare_rank/run_8.toml"),Path("runs_compare_rank/run_9.toml"),Path("runs_compare_rank/run_10.toml"),Path("runs_compare_rank/run_11.toml"),Path("runs_compare_rank/run_12.toml"),Path("runs_compare_rank/run_13.toml"),Path("runs_compare_rank/run_14.toml"),Path("runs_compare_rank/run_15.toml"),Path("runs_compare_rank/run_16.toml"),Path("runs_compare_rank/run_17.toml"),Path("runs_compare_rank/run_18.toml"),Path("runs_compare_rank/run_19.toml"),Path("runs_compare_rank/run_20.toml"),Path("runs_compare_rank/run_21.toml"),Path("runs_compare_rank/run_22.toml"),Path("runs_compare_rank/run_23.toml"),Path("runs_compare_rank/run_24.toml"),Path("runs_compare_rank/run_25.toml"),Path("runs_compare_rank/run_26.toml"),Path("runs_compare_rank/run_27.toml"),Path("runs_compare_rank/run_28.toml"),Path("runs_compare_rank/run_29.toml"),Path("runs_compare_rank/run_30.toml")]
    ranks_history = []
    epochs_global = 0
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

        epochs_global = np.linspace(1, len(layer_ranks[0]), len(layer_ranks[0]))
        ranks_history.append(layer_ranks)

        print(f"-> Training for {tomlFile} finished. Plot saved as 'rank_and_accuracy_plot.png'.")


    nplot = 100
    num_layers = len(ranks_history[-1])
    num_runs = len(ranks_history)

    # Initialize mean and variance containers
    layer_ranks_mu = [np.zeros_like(ranks_history[0][i], dtype=np.float64) for i in range(num_layers)]
    layer_ranks_var = [np.zeros_like(ranks_history[0][i], dtype=np.float64) for i in range(num_layers)]

    # Accumulate sums for mean and squared differences for variance
    for layer_ranks in ranks_history:
        for i in range(num_layers):
            layer_ranks_mu[i] += np.array(layer_ranks[i]) / num_runs

    # Second pass to compute variance
    for layer_ranks in ranks_history:
        for i in range(num_layers):
            diff = np.array(layer_ranks[i]) - layer_ranks_mu[i]
            layer_ranks_var[i] += (diff ** 2) / num_runs

    fig, ax1 = plt.subplots()
    epochs = np.linspace(1, nplot, nplot)
    for i in range(num_layers):
        mu = layer_ranks_mu[i][:nplot]
        std = np.sqrt(layer_ranks_var[i][:nplot])
        
        ax1.plot(epochs, mu, label=f"Layer {i+1}")
        ax1.fill_between(epochs, mu - std, mu + std, alpha=0.2)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Rank")
    ax1.grid(True)
    ax1.legend(loc='upper right')

    fig.tight_layout()

    # Save the figure
    plt.savefig("rank_plot.png", dpi=300)

    # Show the plot
    plt.show()