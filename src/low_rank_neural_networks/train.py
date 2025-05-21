import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from .networks import Network
import numpy as np


class Trainer:
    def __init__(self, netArchitecture, modelDir, logName,
                 confName, trainLoader, testLoader):
        """Constructs trainer which manages and trains neural network
        Args:
            netArchitecture: Dictionary of the network architecture.
                             Needs keys 'type' and 'dims'.
                             Low-rank layers need key 'rank'.
            trainLoader: loader for training data
            testLoader: loader for test data
        """
        # Set the device (GPU or CPU)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self._model = Network(netArchitecture).to(self._device)

        # Find all IDs of dynamical low-rank layers, since these layer require two steps
        self._dlrLayerIds = [index for index, layer in enumerate(netArchitecture)
                             if layer['type'] == 'lowRank' or layer['type'] == 'augmentedLowRank']

        # store train and test data
        self.trainLoader = trainLoader
        self.testLoader = testLoader

        # best test accuracy
        self._testAcc = 0
        self._testAccs = [] # evolution of test accuracy

        # create model directory from log file location
        self._modelDir = modelDir

        # Configure logging
        logFileName = Path(modelDir) / Path(logName)
        self._logger = logging.getLogger(logName)
        handler = logging.FileHandler(logFileName, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        # write config file to logger
        # Open the file and read its content
        with open(confName, 'r') as file:
            file_content = file.read()

        # Log the file content
        self._logger.info(f"Settings defined in {confName}:\n{file_content}")

    def train(self, numEpochs, learningRate, RobbinsMonro=False ):
        """Trains neural network for specified number of epochs
           with specified learning rate
        Args:
            numEpochs: number of epochs for training
            learningRate: learning rate for optimization method
        """

        #self._ranksOverEpoch = np.zeros((len(self._dlrLayerIds), numEpochs))

        self._ranksOverEpoch = [[] for _ in range(len(self._dlrLayerIds))]

        # Define the loss function and optimizer.
        # Optimizer is only needed to set all gradients to zero.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=learningRate)
        t = 0
        initial_lr = learningRate
        tMR = 0 # starting point to reduce lr

        # Training loop
        for epoch in range(numEpochs):
            self._model.train()
            t += 1 # count up iteration counter
            for batchIdx, (data, exactLabels) in enumerate(self.trainLoader):
                if RobbinsMonro and t > tMR:
                    learningRate = initial_lr / ((t-tMR)**1 + 1)
                data = data.to(self._device)
                exactLabels = exactLabels.to(self._device)

                # Forward pass 
                outputs = self._model(data)
                loss = criterion(outputs, exactLabels)

                # Backward to calculate gradients of parameters
                optimizer.zero_grad()
                loss.backward()

                # update entire network without low-rank coefficients
                self._model.step(learningRate)

                # the following part only needed for dynamical low-rank layers
                for i in range(1):
                    # Forward pass
                    outputs = self._model(data)
                    loss = criterion(outputs, exactLabels)

                    # Backward to calculate gradients of coefficients
                    optimizer.zero_grad()
                    loss.backward()

                    # update coefficients in low-rank layers
                    for i in self._dlrLayerIds:
                        self._model._layers[i].step(learningRate, "coefficients")

                # update coefficients in low-rank layers
                for i in self._dlrLayerIds:
                    self._model._layers[i].step(learningRate, "truncate")


                # write out ranks
                self._rankCurrent = []
                for i in self._dlrLayerIds:
                    self._ranksOverEpoch[i].append(self._model._layers[i]._S.shape[0])
                    self._rankCurrent.append(self._model._layers[i]._S.shape[0])

                # print progress
                if (batchIdx + 1) % 100 == 0:

                    print(f"Epoch [{epoch+1}/{numEpochs}], Step [{batchIdx+1}/{len(self.trainLoader)}], Ranks {self._rankCurrent}, Loss: {loss.item():.4f}")

                

            # evaluate model on test date
            self.testModel()

        self._logger.info(f'Best accuracy of architecture is {self._testAcc}%')
        return self._ranksOverEpoch, self._testAccs

    def testModel(self):
        """Prints the model's accuracy on the test data
        """
        # Test the model
        self._model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, exactLabels in self.testLoader:
                data = data.to(self._device)
                exactLabels = exactLabels.to(self._device)

                outputs = self._model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += exactLabels.size(0)
                correct += (predicted == exactLabels).sum().item()

            accuracy = 100 * correct / total
            print(f"Accuracy of the network on the test images: {accuracy}%")
            self._logger.info(f"Accuracy of the network on the test images: {accuracy}%, Ranks {self._rankCurrent}")

            self._testAccs.append(accuracy)

            if self._testAcc < accuracy:
                self._testAcc = accuracy
                self._model.storeParameters(self._modelDir)
