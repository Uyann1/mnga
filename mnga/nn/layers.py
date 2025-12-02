import numpy as np
from .module import Module
from ..autograd import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
        
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias

    def forward(self, x):
        return x @ self.weight + self.bias
