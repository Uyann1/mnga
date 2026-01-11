
import numpy as np
from mnga.autograd import Tensor, log, clamp, softmax, relu
from mnga.nn import Module, Sequential, Linear, ReLU
from mnga.nn.layers import NoisyLinear

def init_layer_uniform(shape, init_w: float = 3e-3):
    return np.random.uniform(-init_w, init_w, shape)

class DQNNetwork(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layers = Sequential(
            Linear(in_dim, 128, weight_init="he_normal"),
            ReLU(),
            Linear(128, 128, weight_init="he_normal"),
            ReLU(),
            Linear(128, out_dim, weight_init=init_layer_uniform)
        )
    def forward(self, x):
        return self.layers(x)

class DuelingNetwork(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # shared trunk
        self.feature = Sequential(
            Linear(in_dim, 128,  weight_init="he_normal"),
            ReLU(),
        )
        # value stream V(s)
        self.value = Sequential(
            Linear(128, 128, weight_init="he_normal"),
            ReLU(),
            Linear(128, 1,      weight_init=init_layer_uniform),
        )
        # advantage stream A(s, a)
        self.advantage = Sequential(
            Linear(128, 128, weight_init="he_normal"),
            ReLU(),
            Linear(128, out_dim, weight_init=init_layer_uniform),
        )

    def forward(self, x):
        # ensure batch dimension
        if x.ndim == 1:
            x = x.reshape(1, -1)
        feat = self.feature(x)
        v = self.value(feat)                  # (batch, 1)
        a = self.advantage(feat)              # (batch, out_dim)
        a_mean = a.mean(axis=1, keepdims=True)
        q = v + (a - a_mean)
        return q

class CategoricalNetwork(Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support
    ):
        super().__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        # Simple feed-forward structure in your framework
        self.layers = Sequential(
            Linear(in_dim, 128), 
            ReLU(),
            Linear(128, 128), 
            ReLU(), 
            Linear(128, out_dim * atom_size)
        )

    def dist(self, x):
        """Get distribution for atoms."""
        # Output shape: (batch_size, out_dim * atom_size)
        q_atoms = self.layers(x)
        
        # Reshape to (batch_size, out_dim, atom_size) using your Tensor view/reshape
        q_atoms = q_atoms.view(-1, self.out_dim, self.atom_size)
        
        # Apply softmax across the atom dimension (the last axis)
        dist = softmax(q_atoms, axis=-1)
        
        # Use your framework's clamp to avoid numerical instability (NaNs)
        dist = dist.clamp(min_value=1e-3)
        
        return dist

    def forward(self, x):
        """Forward method implementation calculating Expected Value Q."""
        # Get distribution p(s, a, z)
        dist = self.dist(x)
        
        # Q(s, a) = Σ p_i * z_i 
        # (batch_size, out_dim, atom_size) * (atom_size,) -> (batch_size, out_dim)
        q = (dist * self.support).sum(axis=2)
        
        return q

class RainbowNetwork(Module):
    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support):
        super().__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # Feature Layer: shared feature extraction
        self.feature_layer = Sequential(
            Linear(in_dim, 128),
            ReLU(),
        )
        
        # Advantage Stream: Noisy Layers
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # Value Stream: Noisy Layers
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)
    
    def dist(self, x):
        # Calculates the probabilty distribution over atoms.
        feature = self.feature_layer(x)

        # Streams
        adv_hid = relu(self.advantage_hidden_layer(feature))
        val_hid = relu(self.value_hidden_layer(feature))

        # Advantage: (batch_size, out_dim * atom_size) -> (batch_size, out_dim, atom_size)
        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)

        # Value: (batch_size, atom_size) -> (batch_size, 1, atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)

        # Dueling Logic: Q_atoms = V + (A - mean(A))
        q_atoms = value + (advantage - advantage.mean(axis=1, keepdims=True))  # (batch_size, out_dim, atom_size)

        # Categorical Logic: Softmax across the atom dimension
        # Clamp to avoid 0.0 or 1.0 which causes NaN in log operations
        dist = softmax(q_atoms, axis=-1)
        dist = clamp(dist, min_value=1e-3)
        return dist  # (batch_size, out_dim, atom_size) -> p(s, a, z)

    def forward(self, x):
        # Get distribution p(s, a)
        dist = self.dist(x)

        # Q(s, a) = Expected Value = sum(p_i * z_ i)
        # (batch_size, actions, atoms) * (atoms,) -> (batch_size, actions)
        q = (dist * self.support).sum(axis=2)
        return q
    
    def reset_noise(self):
        # Reset the learnable noise in all NoisyLinear layers.
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
