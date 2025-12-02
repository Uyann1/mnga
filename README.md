# mnga: Minimal NumPy Gradient Agents

**mnga** is a lightweight, pure NumPy-based Reinforcement Learning framework designed for educational purposes and rapid prototyping of classic control algorithms. It provides clean, readable implementations of popular RL agents without the overhead of heavy deep learning libraries.

## Features

-   **Pure NumPy**: Core logic implemented entirely in NumPy.
-   **Classic Control Focus**: Optimized for environments like CartPole, MountainCar, and Acrobot.
-   **Modular Design**: Easy to extend with new agents, policies, and environments.
-   **Algorithms**: Includes implementations of:
    -   DQN / Double DQN
    -   SARSA
    -   REINFORCE
    -   A2C
    -   PPO

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/mnga.git
cd mnga
pip install -e .
```

## Usage

### Training an Agent

To train a PPO agent on CartPole:

```bash
python examples/train_cartpole_ppo.py
```

### Creating a Custom Agent

```python
import numpy as np
from mnga.rl.agent import Agent

class MyAgent(Agent):
    def act(self, state):
        return np.random.choice([0, 1])
```

## Structure

-   `mnga/`: Core framework code.
    -   `optim/`: Optimizers (SGD, Adam, etc.).
    -   `nn/`: Neural network layers and activations.
    -   `rl/`: RL agents and utilities.
-   `examples/`: Training scripts for various environments.

## License

MIT License
