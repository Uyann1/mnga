import numpy as np
from .module import Module
from ..autograd import Tensor

from . import init

class Linear(Module):
    def __init__(self, in_features, out_features, weight_init=None, bias_init=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.bias_init = bias_init
        
        self.reset_parameters()

    def reset_parameters(self):
        # Weight Initialization
        if self.weight_init is None:
            # Default: He Uniform
            weight_data = init.he_uniform((self.out_features, self.in_features))
        elif isinstance(self.weight_init, str):
            if not hasattr(init, self.weight_init):
                 raise ValueError(f"Unknown initialization method: {self.weight_init}")
            weight_data = getattr(init, self.weight_init)((self.out_features, self.in_features))
        elif callable(self.weight_init):
            weight_data = self.weight_init((self.out_features, self.in_features))
            if not isinstance(weight_data, np.ndarray):
                 raise TypeError("Custom weight initialization must return a numpy array")
        else:
            raise TypeError("weight_init must be None, a string, or a callable")

        self.weight = Tensor(weight_data, requires_grad=True)
        
        # Bias Initialization
        if self.bias_init is None:
             # Default: Zeros
            bias_data = init.zeros((self.out_features,))
        elif isinstance(self.bias_init, str):
             if not hasattr(init, self.bias_init):
                 raise ValueError(f"Unknown initialization method: {self.bias_init}")
             bias_data = getattr(init, self.bias_init)((self.out_features,))
        elif callable(self.bias_init):
            bias_data = self.bias_init((self.out_features,))
            if not isinstance(bias_data, np.ndarray):
                 raise TypeError("Custom bias initialization must return a numpy array")
        else:
             raise TypeError("bias_init must be None, a string, or a callable")

        self.bias = Tensor(bias_data, requires_grad=True)

    def forward(self, x):
        return x @ self.weight.T + self.bias


class NoisyLinear(Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters (mean and sigma)
        # we initialize them as empty Tensors; reset_parameters will set them
        self.weight_mu = Tensor(np.empty((out_features, in_features)), requires_grad=True)
        self.weight_sigma = Tensor(np.empty((out_features, in_features)), requires_grad=True)

        self.bias_mu = Tensor(np.empty(out_features), requires_grad=True)
        self.bias_sigma = Tensor(np.empty(out_features), requires_grad=True)

        # Noise Buffers (Static noise samples - no gradients needed)
        self.weight_epsilon = Tensor(np.zeros((out_features, in_features)), requires_grad=False)
        self.bias_epsilon = Tensor(np.zeros(out_features), requires_grad=False)

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
       # Initializes mu and sigma according to Factorized Gaussian Noise specs.
        mu_range = 1 / np.sqrt(self.in_features)

        # Initialize Mu(means) using uniform distribution
        self.weight_mu.data[:] = np.random.uniform(-mu_range, mu_range, (self.out_features, self.in_features))
        self.bias_mu.data[:] = np.random.uniform(-mu_range, mu_range, (self.out_features,))

        # Initialize Sigma (Noise intensity)
        self.weight_sigma.data.fill(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        # Generates scaled noise: f(x) = sign(x) * sqrt(|x|)
        x = np.random.standard_normal(size)
        return (np.sign(x) * np.sqrt(np.abs(x))).astype(np.float32)
    
    def reset_noise(self): 
        # Generates new factorized Gaussian noise for the weight and bias.
        epsilon_int = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product to get weight noise
        self.weight_epsilon.data[:] = np.outer(epsilon_out, epsilon_int)
        self.bias_epsilon.data[:] = epsilon_out

    def forward(self, x):
        """
        Forward pass: y = x @ (mu_w + sigma_w * eps_w).T + (mu_b + sigma_b * eps_b)
        
        """
        
        curr_weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        curr_bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return x @ curr_weight.T + curr_bias 