import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import copy

class ActivationFactory:
    '''
    Factory to simplify the creation of activation functions for each layer
    '''
    def __init__(self) -> None:
        '''Constructor initializes dictionary with standard activations'''
        self._activationTypes = {'relu': nn.ReLU(), 'linear': nn.Identity(),
                                 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(),
                                 'leakyRelu': nn.LeakyReLU(), 'swish': nn.SiLU(),
                                 'softplus': nn.Softplus()}

    def register(self, key, name):
        '''Register function to add further entries to factory dictionary
        Args:
            key: user key to create activation
            name: corresponding name of activation function in torch
        '''
        self._activationTypes[key] = name

    def __call__(self, key):
        '''Create activation function from user key
        Args:
            key: user key to create activation
        '''
        return self._activationTypes[key]


class LayerFactory:
    '''Factory to simplify the creation of layers'''
    def __init__(self) -> None:
        '''Constructor initializes dictionary with standard layers'''
        self._layerTypes = {'dense': DenseLayer,
                            'vanillaLowRank': VanillaLowRankLayer,
                            'lowRank': DynamicalLowRankLayer,
                            'augmentedLowRank': AugmentedDynamicalLowRankLayer,
                            'fineTuneLowRank': FTLayer,
                            'Burer-Monteiro': BMLayer}

    def register(self, key, name):
        '''Register function to add further entries to factory dictionary
        Args:
            key: user key to create layers
            name: corresponding name of Layer class
        '''
        self._layerTypes[key] = name

    def __call__(self, layer):
        '''Create layers from user key
        Args:
            key: user key to create layer
        '''
        key = layer["type"]
        return self._layerTypes[key](layer)


class Layer(nn.Module):
    def __init__(self, layerParams):
        """Constructs a layer of a general form
        Args:
            layerParams: dictionary with dimensions, activation
        """
        # construct parent class nn.Module
        super(Layer, self).__init__()

        # define bias as trainable parameter
        self._bias = nn.Parameter(torch.randn(layerParams['dims'][1]))

        if "loadFile" in layerParams:
            try:
                file = Path(layerParams["loadFile"] + '_b.pth')
                datab = torch.load(file)
                if self._bias.shape == datab.shape:
                    self._bias.data = datab
                else:
                    raise ValueError("Dimension in b not matching.")
            except ValueError as err:
                print(f"Error: Encountered error while reading bias from file - {err}")
                exit()
            except FileNotFoundError:
                print(f"Error: The provided file name does not exist - {file}")
                exit()

        # construct activation function
        activation = ActivationFactory()
        self._sigma = activation(layerParams['activation'])

    @property
    def b(self):
        """Property to return bias"""
        return self._bias


class DenseLayer(Layer):
    def __init__(self, layerParams):
        """Constructs a dense layer of the form W*x + b, where W
           is the weight matrix and b is the bias vector.
        Args:
            layerParams: dictionary with dimensions, activation
        """
        # construct parent class nn.Module
        super().__init__(layerParams)
        # define weights as trainable parameter
        self._W = nn.Parameter(torch.randn(layerParams['dims'][0], layerParams['dims'][1]))
        
        # load weights if input location is specified
        if "loadFile" in layerParams:
            try:
                file = Path(layerParams["loadFile"] + '_W.pth')
                dataW = torch.load(file)
                if self._W.shape == dataW.shape:
                    self._W.data = dataW
                else:
                    raise ValueError("Dimension in W not matching.")
            except ValueError as err:
                print(f"Error: Encountered error while reading weight from file - {err}")
                exit()
            except FileNotFoundError:
                print(f"Error: The provided file name does not exist - {file}")
                exit()

    def forward(self, x):
        """Returns the output of the layer.
           The formula implemented is output = W*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        out = torch.matmul(x, self._W)
        return self._sigma(out + self._bias)

    @torch.no_grad()
    def step(self, learning_rate):
        """Performs a steepest descend training update on weights and biases
        Args:
            learning_rate: learning rate for training
        """
        self._W.data = self._W - learning_rate * self._W.grad
        self._bias.data = self._bias - learning_rate * self._bias.grad
    
    @torch.no_grad()
    def storeParameters(self, namePre, name):
        """Stores parameters of dense layer
        Args:
            namePre: name for storing weights and biases
        """
        torch.save(self._W, namePre / Path(name + '_W.pth'))
        torch.save(self._bias, namePre / Path(name + '_b.pth'))

    @property
    def W(self):
        """Property to return weights"""
        return self._W


class LowRankLayer(Layer):
    def __init__(self, layerParams):
        """Constructs a parent class low-rank layer of the
           form U*S*V'*x + b, where U, S, V represent the facorized
           weight W and b is the bias vector.
           Step function for training is not defined
        Args:
            layerParams: dictionary with input
                            and output dimension as well as rank
        """
        # construct parent class nn.Module
        super().__init__(layerParams)
        # initializes factorized weight
        self._U = nn.Parameter(torch.randn(layerParams['dims'][0], layerParams['rank']))
        self._S = nn.Parameter(torch.randn(layerParams['rank'], layerParams['rank']))
        self._V = nn.Parameter(torch.randn(layerParams['dims'][1], layerParams['rank']))

        # if load file provided, weights are loaded from file
        if "loadFile" in layerParams:
            try:
                dataU = torch.load(Path(layerParams["loadFile"] + '_U.pth'))
                self._U = nn.Parameter(dataU)

                dataS = torch.load(Path(layerParams["loadFile"] + '_S.pth'))
                self._S = nn.Parameter(dataS)

                dataV = torch.load(Path(layerParams["loadFile"] + '_V.pth'))
                self._V = nn.Parameter(dataV)

                dataBias = torch.load(Path(layerParams["loadFile"] + '_b.pth'))
                self._bias = nn.Parameter(dataBias)
            except ValueError as err:
                print(f"Error: Encountered error while reading low-rank layer from file - {err}")
                exit()

            except FileNotFoundError:
                file = Path(layerParams["loadFile"] + '_{U,S,V}}.pth')
                print(f"Error: The provided file name does not exist - {file}")
                exit()
        else:
            # ensure that U and V are orthonormal
            self._U.data, _ = torch.linalg.qr(self._U, 'reduced')
            self._V.data, _ = torch.linalg.qr(self._V, 'reduced')

        # set rank and truncation tolerance
        self._r = layerParams['rank']

    def forward(self, x):
        """Returns the output of the layer.
           The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns:
            output of layer
        """
        xU = torch.matmul(x, self._U)
        xUS = torch.matmul(xU, self._S)
        out = torch.matmul(xUS, self._V.T)
        return self._sigma(out + self._bias)

    @torch.no_grad()
    def storeParameters(self, namePre, name):
        """Stores parameters of vanilla low-rank layer
        Args:
            namePre: name for storing low-rank weights and biases
        """
        torch.save(self._U, namePre / Path(name + '_U.pth'))
        torch.save(self._V, namePre / Path(name + '_V.pth'))
        torch.save(self._S, namePre / Path(name + '_S.pth'))
        torch.save(self._bias, namePre / Path(name + '_b.pth'))

    @property
    def U(self):
        """Property to return factorized weight U"""
        return self._U
    
    @property
    def V(self):
        """Property to return factorized weight V"""
        return self._V
    
    @property
    def S(self):
        """Property to return factorized weight S"""
        return self._S


class VanillaLowRankLayer(LowRankLayer):
    """Constructs a vanilla low-rank layer of the form U*S*V'*x + b, where 
       U, S, V represent the facorized weight W and b is the bias vector
    Args:
        layerParams: dictionary with input and output dimension as well as rank
    """
    def __init__(self, layerParams):
        super().__init__(layerParams)

    @torch.no_grad()
    def step(self, learning_rate):
        """Performs a steepest descend training update on factorized weight and bias
        Args:
            learning_rate: learning rate for training
        """
        self._U.data = self._U - learning_rate * self._U.grad
        self._V.data = self._V - learning_rate * self._V.grad
        self._S.data = self._S - learning_rate * self._S.grad
        self._bias.data = self._bias - learning_rate * self._bias.grad

class BMLayer(LowRankLayer):
    """Constructs a vanilla low-rank layer of the form U*S*V'*x + b, where 
       U, S, V represent the facorized weight W and b is the bias vector
    Args:
        layerParams: dictionary with input and output dimension as well as rank
    """
    def __init__(self, layerParams):
        """Constructs a parent class low-rank layer of the
           form U*S*V'*x + b, where U, S, V represent the facorized
           weight W and b is the bias vector.
           Step function for training is not defined
        Args:
            layerParams: dictionary with input
                            and output dimension as well as rank
        """
        # construct parent class nn.Module
        super().__init__(layerParams)
        # initializes factorized weight
        self._U = nn.Parameter(torch.randn(layerParams['dims'][0], layerParams['rank']))
        self._V = nn.Parameter(torch.randn(layerParams['dims'][1], layerParams['rank']))

        # if load file provided, weights are loaded from file
        if "loadFile" in layerParams:
            try:
                dataU = torch.load(Path(layerParams["loadFile"] + '_U.pth'))
                self._U = nn.Parameter(dataU)

                dataV = torch.load(Path(layerParams["loadFile"] + '_V.pth'))
                self._V = nn.Parameter(dataV)

                dataBias = torch.load(Path(layerParams["loadFile"] + '_b.pth'))
                self._bias = nn.Parameter(dataBias)
            except ValueError as err:
                print(f"Error: Encountered error while reading low-rank layer from file - {err}")
                exit()

            except FileNotFoundError:
                file = Path(layerParams["loadFile"] + '_{U,V}}.pth')
                print(f"Error: The provided file name does not exist - {file}")
                exit()

        # set rank and truncation tolerance
        self._r = layerParams['rank']

    def forward(self, x):
        """Returns the output of the layer.
           The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns:
            output of layer
        """
        xU = torch.matmul(x, self._U)
        out = torch.matmul(xU, self._V.T)
        return self._sigma(out + self._bias)

    @torch.no_grad()
    def storeParameters(self, namePre, name):
        """Stores parameters of vanilla low-rank layer
        Args:
            namePre: name for storing low-rank weights and biases
        """
        torch.save(self._U, namePre / Path(name + '_U.pth'))
        torch.save(self._V, namePre / Path(name + '_V.pth'))
        torch.save(self._bias, namePre / Path(name + '_b.pth'))

    @property
    def U(self):
        """Property to return factorized weight U"""
        return self._U
    
    @property
    def V(self):
        """Property to return factorized weight V"""
        return self._V

    @torch.no_grad()
    def step(self, learning_rate):
        """Performs a steepest descend training update on factorized weight and bias
        Args:
            learning_rate: learning rate for training
        """
        self._U.data = self._U - learning_rate * self._U.grad
        self._V.data = self._V - learning_rate * self._V.grad
        self._bias.data = self._bias - learning_rate * self._bias.grad

class FTLayer(LowRankLayer):
    """Constructs a vanilla low-rank layer of the form U*S*V'*x + b, where 
       U, S, V represent the facorized weight W and b is the bias vector
    Args:
        layerParams: dictionary with input and output dimension as well as rank
    """
    def __init__(self, layerParams):
        super().__init__(layerParams)

    @torch.no_grad()
    def step(self, learning_rate):
        """Performs a steepest descend training update on factorized weight and bias
        Args:
            learning_rate: learning rate for training
        """
        self._S.data = self._S - learning_rate * self._S.grad
        self._bias.data = self._bias - learning_rate * self._bias.grad


class DynamicalLowRankLayer(LowRankLayer):
    def __init__(self, layerParams):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            layerParams: dictionary with input, output dimension and rank
        """
        # construct parent class nn.Module
        super().__init__(layerParams)
        self._rCurrent = self._U.data.shape[1]

    @torch.no_grad()
    def step(self, learning_rate, dlrt_step="basis"):
        """Performs a steepest descend training update on specified
           low-rank factors according to
           the dynamical low-rank training method
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'basis' (default) or 'coefficients'
        """

        if dlrt_step == "basis":
            # perform K-step
            U0 = self._U.data
            V0 = self._V.data
            S0 = self._S.data
            K = torch.matmul(U0, S0)
            dK = torch.matmul(self._U.grad, S0) + torch.matmul(U0, self._S.grad)
            K = K - learning_rate * dK
            self._U.data, _ = torch.linalg.qr(K, 'reduced')

            # perform L-step
            L = torch.matmul(V0, S0.T)
            dL = torch.matmul(self._V.grad, S0.T) + torch.matmul(V0, self._S.grad.T)
            L = L - learning_rate * dL
            self._V.data, _ = torch.linalg.qr(L, 'reduced')

            # project coefficients
            M = self._U.data.T @ U0
            N = V0.T @ self._V.data
            self._S.data = M @ self._S @ N

            # update bias
            self._bias.data = self._bias - learning_rate * self._bias.grad

            # deactivate gradient tape for basis
            self._U.requires_grad_ = False
            self._V.requires_grad_ = False

        elif dlrt_step == "coefficients":
            self._S.data = self._S - learning_rate * self._S.grad

            # activate gradient tape for basis
            self._U.requires_grad_ = True
            self._V.requires_grad_ = True
        else:
            print("Wrong step defined: ", dlrt_step)

class AugmentedDynamicalLowRankLayer(LowRankLayer):
    def __init__(self, layerParams, init_compression=0.2):
        """Constructs a rank-adaptive low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            layerParams: dictionary with input, output dimension and rank
        """
        # Double the rank for initialization to make sure large enough matrices are allocated
        self._rMax = layerParams["rank"] 
        layerParams["rank"] = int(init_compression * layerParams["rank"])
        # Construct parent class nn.Module
        super().__init__(layerParams)
        layerParams["rank"] = self._rMax

        # Set rank and truncation tolerance
        self._tol = layerParams['tol']

    def forward(self, x):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        xU = torch.matmul(x, self._U)
        xUS = torch.matmul(xU, self._S)
        out = torch.matmul(xUS, self._V.T)
        return self._sigma(out + self._bias)

    @torch.no_grad()
    def step(self, learning_rate, dlrt_step="basis"):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
            adaptive: specifies if fixed-rank or rank-adaptivity is used
        """
        self.adaptive = True

        if dlrt_step == "basis":
            # Perform K-step
            U0 = self._U.data
            V0 = self._V.data
            S0 = self._S.data
            r = S0.shape[0]

            self._Usave = copy.deepcopy(U0)
            self._Vsave = copy.deepcopy(V0)

            K = torch.matmul(U0, S0)
            #dK = torch.matmul(self._U.grad[:,:r], S0)
            Udata, _ = torch.linalg.qr(torch.cat((U0, self._U.grad),1), 'reduced')
            self._U = nn.Parameter(Udata)
            #self._U[:,:r] = U0.data # this is somehow not allowed


            # Perform L-step
            L = torch.matmul(V0, S0.T)
            #dL = torch.matmul(self._V.grad[:,:r], S0.T)
            Vdata, _ = torch.linalg.qr(torch.cat((V0, self._V.grad),1), 'reduced')
            self._V = nn.Parameter(Vdata)
            #self._V[:,:r] = V0.data # this is somehow not allowed

            M = self._U.T @ U0
            N = V0.T @ self._V

            self._S = nn.Parameter( M @ self._S @ N )

            # Update bias
            self._bias.data = self._bias - learning_rate * self._bias.grad

            # Deactivate gradient tape for basis
            self._U.requires_grad_ = False
            self._V.requires_grad_ = False

        elif dlrt_step == "coefficients":
            self._S.data = self._S - learning_rate * self._S.grad
            self._S._grad.zero_()
        elif dlrt_step == "truncate":
            # Truncate to new rank
            self._truncate(10000 * learning_rate**2)

            # Activate gradient tape for basis
            self._U.requires_grad_ = True
            self._V.requires_grad_ = True
        else:
            print("Wrong step defined: ", dlrt_step)

    @torch.no_grad()
    def _truncate(self, tol=-1):
        r = self._S.shape[0]
        P, d, Q = torch.linalg.svd(self._S)

        if tol < 0:
            tol = self._tol * torch.linalg.norm(d)
        else:
            tol *= torch.linalg.norm(d)*0.1
        r1 = r
        for j in range(0, r - 1):
            tmp = torch.linalg.norm(d[j:r - 1])
            if tmp < tol:
                #print(tmp)
                r1 = j
                break

        r1 = min(r1, self._rMax) # Check if rank is bigger than maximal rank
        r1 = int(max(r1, 2))

        # Update s
        self._S = nn.Parameter(torch.diag(d[:r1]))

        # Update u and v
        self._U = nn.Parameter(torch.matmul(self._U, P[:, :r1]))
        self._V = nn.Parameter(torch.matmul(self._V, Q.T[:, :r1]))

        #print(np.linalg.norm(self._U @ (self._U.T @ self._Usave) - self._Usave))