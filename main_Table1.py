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
        RobbinsMonro = settings.get('RobbinsMonro', False)

        architecture = config.get('layer')

        if learningRate is None or architecture is None:
            raise ValueError("The TOML file is missing required fields.")
        
        return learningRate, batchSize, numEpochs, RobbinsMonro, outputDir, logName, architecture

    except FileNotFoundError:
        print(f"Error: File not found - {filePath}")
        exit()
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    #tomlFiles = [Path("runs_compare_restart/dlrt_net_30.toml"), Path("runs_compare_restart/dlrt_net_60.toml"), Path("runs_compare_restart/dlrt_net_100.toml"), Path("runs_compare_restart/dlrt_net_ft_30.toml"),Path("runs_compare_restart/dlrt_net_ft_60.toml"),Path("runs_compare_restart/dlrt_net_ft_100.toml"),Path("runs_compare_restart/dlrt_net_continue_30.toml"),Path("runs_compare_restart/dlrt_net_continue_60.toml"),Path("runs_compare_restart/dlrt_net_continue_100.toml")]
    tomlFiles = [Path("runs_compare_restart/dlrt_net_ft_30.toml"),Path("runs_compare_restart/dlrt_net_ft_60.toml"),Path("runs_compare_restart/dlrt_net_ft_100.toml"),Path("runs_compare_restart/dlrt_net_continue_30.toml"),Path("runs_compare_restart/dlrt_net_continue_60.toml"),Path("runs_compare_restart/dlrt_net_continue_100.toml")]

    print(tomlFiles)
    for tomlFile in tomlFiles:

        learningRate, batchSize, numEpochs, RobbinsMonro, outputDir, logName, architecture = read_configuration(tomlFile)

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
        layer_ranks, test_accs = trainer.train(numEpochs, learningRate, RobbinsMonro)

        print(f"-> Training for {tomlFile} finished.")
