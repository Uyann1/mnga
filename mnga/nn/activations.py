import numpy as np
from .module import Module
from ..autograd import maximum, exp, tanh

class ReLU(Module):
    def forward(self, x):
        return maximum(x, 0)

class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + exp(-x))

class Tanh(Module):
    def forward(self, x):
        return tanh(x)
