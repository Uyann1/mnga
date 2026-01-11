# 🔬 From-Scratch Autograd Engine and Deep Reinforcement Learning Algorithms

This repository contains a minimal, from-scratch implementation of an automatic differentiation engine and a suite of modern Deep Reinforcement Learning algorithms, implemented without using PyTorch or TensorFlow.

The primary goal of this project is **conceptual correctness** and **systems-level understanding**, rather than performance or production deployment.

---

## 📌 Motivation

Most reinforcement learning implementations rely heavily on mature frameworks that abstract away:
- gradient propagation
- computation graph structure
- memory lifetime
- optimizer mechanics

While powerful, these abstractions obscure many critical design decisions.

This project was built to:
- understand how gradients actually flow through a DAG
- reason about graph correctness and memory safety
- implement modern RL algorithms at the algorithmic level, not API level
- explore stability mechanisms (target networks, PER, distributional RL, noisy exploration)

---

## 🧠 Autograd Engine

### Core Design Principles
- Explicit computation graph (DAG)
- Reverse-mode automatic differentiation
- Topological ordering for correct gradient propagation
- Manual memory management via graph pruning
- PyTorch-compatible semantics where applicable

### Key Features
- Tensor abstraction backed by NumPy
- Function-based operator system (`Function.apply`)
- Context objects (`ctx`) for backward propagation
- `no_grad()` context manager
- Safe graph deletion with `retain_graph` semantics
- Gradient accumulation and broadcasting-aware backward passes

The engine enforces the invariant:

> **Every node in the computation graph is visited exactly once during backpropagation.**

This avoids both:
- incorrect gradient truncation
- exponential recomputation in DAGs with shared subgraphs

---

## 🧮 Implemented Operations

| Category | Operations |
|----------|------------|
| **Elementwise** | `add`, `sub`, `mul`, `div`, `neg`, `pow`, `abs` |
| **Reductions** | `sum`, `mean`, `max` |
| **Linear Algebra** | `matmul` (generalized batch support) |
| **Nonlinearities** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax` |
| **Losses** | Huber loss (mean & unreduced for PER), MSE |
| **Shape Ops** | `reshape`, `view`, `transpose` |
| **Indexing** | `getitem` for categorical RL projections |
| **Other** | `exp`, `log`, `clamp`, `maximum` |

Each operation includes:
- a mathematically derived backward pass
- broadcasting-safe gradient handling

---

## 🎮 Reinforcement Learning Algorithms

All RL algorithms are implemented directly on top of the custom autograd engine.

### Value-Based Methods

| Algorithm | Reference |
|-----------|-----------|
| **DQN** | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) — Human-level control through deep reinforcement learning |
| **Double DQN** | [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461) — Reducing overestimation bias |
| **Dueling DQN** | [Wang et al., 2016](https://arxiv.org/abs/1511.06581) — Separating value and advantage estimation |
| **Noisy Networks** | [Fortunato et al., 2018](https://arxiv.org/abs/1706.10295) — Parameterized exploration |
| **PER** | [Schaul et al., 2016](https://arxiv.org/abs/1511.05952) — Prioritized Experience Replay |
| **N-step Returns** | Sutton & Barto — Multi-step bootstrapping |
| **Categorical DQN (C51)** | [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887) — Distributional reinforcement learning |
| **Rainbow DQN** | [Hessel et al., 2018](https://arxiv.org/abs/1710.02298) — Combining multiple DQN improvements |



## 📊 Design Choices

- PER uses importance sampling weights
- Huber loss supports unreduced mode for priority updates
- Graph pruning is explicitly handled to control memory growth


---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mnga.git
cd mnga

# Install in development mode
pip install -e .

```

---

## 📁 Project Structure

```
mnga/
├── mnga/
│   ├── autograd.py      # Core autograd engine
│   ├── nn/              # Neural network modules
│   │   ├── __init__.py
│   │   ├── module.py    # Base Module class
│   │   ├── linear.py    # Linear layer
│   │   ├── activations.py
│   │   └── losses.py
│   ├── optim/           # Optimizers
│   │   └── optimizers.py  # SGD, Adam, RMSprop
│   └── rl/              # RL components
│       ├── agents.py    # All RL agents
│       ├── buffers.py   # Replay buffers
│       └── networks.py  # Agent networks
├── examples/            # Training scripts
├── tests/               # Unit tests
└── README.md
```

---

## 📚 References

This implementation was informed by the following papers:

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature (2015)
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (2016)
- Schaul et al., *Prioritized Experience Replay* (2016)
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning* (2016)
- Bellemare et al., *A Distributional Perspective on Reinforcement Learning* (2017)
- Fortunato et al., *Noisy Networks for Exploration* (2018)
- Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning* (2018)

### Implementation Inspiration

- [rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need)

This repository re-implements the ideas without relying on PyTorch, focusing instead on **correctness** and **interpretability**.

---

## � Contributors

- **Umutcan Uyan** ([@Uyann1](https://github.com/Uyann1))
- **Bahri Uyan** ([@DaTTeBaY0o00](https://github.com/DaTTeBaY0o00))

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
