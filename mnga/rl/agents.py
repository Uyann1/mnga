import numpy as np
import copy
from ..nn import Module, Linear, ReLU, MSELoss, Tanh, Sigmoid
from ..optim import Adam, SGD
from .buffers import ReplayBuffer
from ..autograd import Tensor, log, exp, maximum, tanh

class QNetwork(Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden_dim, action_dim)
        
        self._parameters['fc1'] = self.fc1
        self._parameters['fc2'] = self.fc2
        self._parameters['fc3'] = self.fc3

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, network_cls=QNetwork):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = network_cls(state_dim, action_dim)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.criterion = MSELoss()
        
        self.memory = ReplayBuffer(buffer_size, state_dim)
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = Tensor(state[np.newaxis, :])
        q_values = self.q_network(state)
        return np.argmax(q_values.data)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.train_step()
            
        if done:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        return loss

    def train_step(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = Tensor(states)
        next_states = Tensor(next_states)
        rewards = Tensor(rewards)
        dones = Tensor(dones.astype(np.float32))
        
        next_q_values = self.target_network(next_states)
        max_next_q_values = np.max(next_q_values.data, axis=1, keepdims=True)
        target_q_values = rewards + self.gamma * Tensor(max_next_q_values) * (Tensor(1.0) - dones)
        
        current_q_values = self.q_network(states)
        
        actions_one_hot = np.zeros((self.batch_size, self.action_dim))
        actions_one_hot[np.arange(self.batch_size), actions.flatten()] = 1
        actions_one_hot = Tensor(actions_one_hot)
        
        q_taken = (current_q_values * actions_one_hot).sum(axis=1, keepdims=True)
        
        loss = self.criterion(q_taken, target_q_values)
        
        self.q_network.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data.item()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.q_network)

class DoubleDQNAgent(DQNAgent):
    def train_step(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = Tensor(states)
        next_states = Tensor(next_states)
        rewards = Tensor(rewards)
        dones = Tensor(dones.astype(np.float32))
        
        next_q_values_online = self.q_network(next_states)
        next_actions = np.argmax(next_q_values_online.data, axis=1)
        
        next_q_values_target = self.target_network(next_states)
        
        next_actions_one_hot = np.zeros((self.batch_size, self.action_dim))
        next_actions_one_hot[np.arange(self.batch_size), next_actions] = 1
        next_actions_one_hot = Tensor(next_actions_one_hot)
        
        max_next_q_values = (next_q_values_target * next_actions_one_hot).sum(axis=1, keepdims=True)
        
        target_q_values = rewards + self.gamma * max_next_q_values * (Tensor(1.0) - dones)
        target_q_values = target_q_values.detach()
        
        current_q_values = self.q_network(states)
        
        actions_one_hot = np.zeros((self.batch_size, self.action_dim))
        actions_one_hot[np.arange(self.batch_size), actions.flatten()] = 1
        actions_one_hot = Tensor(actions_one_hot)
        
        q_taken = (current_q_values * actions_one_hot).sum(axis=1, keepdims=True)
        
        loss = self.criterion(q_taken, target_q_values)
        
        self.q_network.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data.item()

class SARSAAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_action = None

    def act(self, state):
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
            
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = Tensor(state[np.newaxis, :])
        q_values = self.q_network(state)
        return np.argmax(q_values.data)

    def step(self, state, action, reward, next_state, done):
        # Determine next action (on-policy)
        if np.random.rand() < self.epsilon:
            next_action = np.random.randint(self.action_dim)
        else:
            next_state_t = Tensor(next_state[np.newaxis, :])
            q_values = self.q_network(next_state_t)
            next_action = np.argmax(q_values.data)
        
        self.next_action = next_action # Store for next act call
        
        state_t = Tensor(state[np.newaxis, :])
        next_state_t = Tensor(next_state[np.newaxis, :])
        reward_t = Tensor(np.array([[reward]]))
        done_t = Tensor(np.array([[done]], dtype=np.float32))
        
        # Target: r + gamma * Q_target(s', a')
        next_q_values = self.target_network(next_state_t)
        target_q_val = next_q_values[0, next_action]
        
        target = reward_t + self.gamma * target_q_val * (Tensor(1.0) - done_t)
        target = target.detach()
        
        # Current: Q(s, a)
        current_q_values = self.q_network(state_t)
        current_q_val = current_q_values[0, action]
        
        loss = (current_q_val - target) ** 2
        
        self.q_network.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if done:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.next_action = None # Reset
            
        return loss.data.item()

class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_dim, action_dim)
        
        self._parameters['fc1'] = self.fc1
        self._parameters['fc2'] = self.fc2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # Softmax
        # Exp and sum
        exps = exp(x - np.max(x.data, axis=1, keepdims=True)) # Stable softmax
        return exps / exps.sum(axis=1, keepdims=True)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        state = Tensor(state[np.newaxis, :])
        probs = self.policy_net(state)
        probs_np = probs.data.flatten()
        action = np.random.choice(self.action_dim, p=probs_np)
        
        # Store log prob of chosen action
        self.log_probs.append(log(probs[0, action]))
        return action

    def step(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        
        loss = None
        if done:
            loss = self.train_step()
            self.log_probs = []
            self.rewards = []
        return loss

    def train_step(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = Tensor(np.array(returns))
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.data.std() + 1e-9)
        
        policy_loss = Tensor(0.0)
        for log_prob, R in zip(self.log_probs, returns.data):
            policy_loss = policy_loss - log_prob * Tensor(R)
            
        self.policy_net.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.data.item()

class ActorCriticNetwork(Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.relu1 = ReLU()
        
        # Actor
        self.actor_fc = Linear(hidden_dim, action_dim)
        
        # Critic
        self.critic_fc = Linear(hidden_dim, 1)
        
        self._parameters['fc1'] = self.fc1
        self._parameters['actor_fc'] = self.actor_fc
        self._parameters['critic_fc'] = self.critic_fc

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        # Actor
        logits = self.actor_fc(x)
        exps = exp(logits - np.max(logits.data, axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        
        # Critic
        value = self.critic_fc(x)
        
        return probs, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.network.parameters(), lr=lr)
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def act(self, state):
        state = Tensor(state[np.newaxis, :])
        probs, value = self.network(state)
        probs_np = probs.data.flatten()
        action = np.random.choice(self.action_dim, p=probs_np)
        
        self.log_probs.append(log(probs[0, action]))
        self.values.append(value)
        self.entropies.append(-(probs * log(probs)).sum())
        
        return action

    def step(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        
        loss = None
        if done:
            # Bootstrap value is 0 for terminal state
            loss = self.train_step(next_value=0)
            self.reset_memory()
        return loss
        
    def reset_memory(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def train_step(self, next_value=0):
        R = next_value
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = Tensor(np.array(returns).reshape(-1, 1))
        
        policy_loss = Tensor(0.0)
        value_loss = Tensor(0.0)
        entropy_loss = Tensor(0.0)
        
        for log_prob, value, R, entropy in zip(self.log_probs, self.values, returns.data, self.entropies):
            advantage = Tensor(R) - value
            
            policy_loss = policy_loss - log_prob * advantage.detach()
            value_loss = value_loss + advantage ** 2
            entropy_loss = entropy_loss - entropy
            
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        self.network.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.data.item()

class GaussianPolicy(Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.relu1 = ReLU()
        
        self.mean_fc = Linear(hidden_dim, action_dim)
        self.log_std = Tensor(np.zeros((1, action_dim)), requires_grad=True) # Learnable log_std
        
        self.critic_fc = Linear(hidden_dim, 1)
        
        self._parameters['fc1'] = self.fc1
        self._parameters['mean_fc'] = self.mean_fc
        self._parameters['log_std'] = self.log_std
        self._parameters['critic_fc'] = self.critic_fc

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        mean = self.mean_fc(x)
        mean = tanh(mean) * 2.0 # Scale to [-2, 2] for Pendulum, or just tanh
        # Pendulum-v1 action is [-2, 2]. Tanh gives [-1, 1].
        
        value = self.critic_fc(x)
        
        return mean, self.log_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, continuous=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.continuous = continuous
        
        if continuous:
            self.policy = GaussianPolicy(state_dim, action_dim)
        else:
            self.policy = ActorCriticNetwork(state_dim, action_dim)
            
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.policy_old = copy.deepcopy(self.policy)
        
        self.buffer = []

    def act(self, state):
        state = Tensor(state[np.newaxis, :])
        
        if self.continuous:
            mean, log_std, value = self.policy_old(state)
            std = exp(log_std)
            
            # Sample from Gaussian
            # z ~ N(0, 1)
            z = np.random.normal(0, 1, size=mean.shape)
            action = mean.data + std.data * z
            action = np.clip(action, -2.0, 2.0) # Clip for environment
            
            # Compute log prob
            # log N(x; mu, sigma) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2pi)
            # We need to compute this graph-connected for training, but here just value?
            # Wait, act returns log_prob for storage.
            # We can compute it using Tensors to be safe or just numpy if we don't backprop through old_log_prob (which we don't).
            # But we need it for the ratio later.
            
            # Re-compute using Tensors to get value? No, act is inference.
            # But we store log_prob.
            
            var = std ** 2
            log_prob = -0.5 * ((Tensor(action) - mean) ** 2) / var - log_std - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob.sum(axis=1) # Sum over action dim
            
            return action[0], log_prob.data.item(), value.data.item()
            
        else:
            probs, value = self.policy_old(state)
            probs_np = probs.data.flatten()
            action = np.random.choice(self.action_dim, p=probs_np)
            
            log_prob = log(probs[0, action])
            return action, log_prob.data.item(), value.data.item()

    def step(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, log_prob, reward, done, value))
        
        loss = None
        if done: 
            loss = self.train_step()
            self.buffer = []
        return loss

    def train_step(self):
        # Convert buffer to arrays
        states = np.array([x[0] for x in self.buffer])
        actions = np.array([x[1] for x in self.buffer])
        old_log_probs = np.array([x[2] for x in self.buffer])
        rewards = np.array([x[3] for x in self.buffer])
        dones = np.array([x[4] for x in self.buffer], dtype=np.float32)
        values = np.array([x[5] for x in self.buffer])
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = Tensor(np.array(returns).reshape(-1, 1))
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.data.std() + 1e-7)
        
        states = Tensor(states)
        actions = Tensor(actions)
        old_log_probs = Tensor(old_log_probs)
        
        total_loss_val = 0
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            if self.continuous:
                mean, log_std, state_values = self.policy(states)
                std = exp(log_std)
                var = std ** 2
                
                # log prob of actions
                # actions: (N, action_dim)
                # mean: (N, action_dim)
                # log_std: (1, action_dim) broadcasted
                
                # We need to ensure actions is Tensor with correct shape
                if len(actions.shape) == 1:
                    actions = actions.reshape(-1, 1) # Should be (N, action_dim)
                
                log_probs = -0.5 * ((actions - mean) ** 2) / var - log_std - 0.5 * np.log(2 * np.pi)
                log_probs = log_probs.sum(axis=1, keepdims=True) # (N, 1)
                
                dist_entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
                dist_entropy = dist_entropy.sum(axis=1).mean()
                
            else:
                # Evaluating old actions and values
                probs, state_values = self.policy(states)
                
                # Gather log probs
                actions_one_hot = np.zeros((len(actions.data), self.action_dim))
                actions_one_hot[np.arange(len(actions.data)), actions.data.astype(int)] = 1
                actions_one_hot = Tensor(actions_one_hot)
                
                action_probs = (probs * actions_one_hot).sum(axis=1, keepdims=True)
                log_probs = log(action_probs)
                
                dist_entropy = -(probs * log(probs)).sum(axis=1).mean()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            # log_probs and old_log_probs should be (N, 1)
            # old_log_probs might be (N,) from buffer, reshape
            old_log_probs_t = old_log_probs.reshape(-1, 1)
            
            ratios = exp(log_probs - old_log_probs_t)
            
            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            
            # clamp(x, min, max) = minimum(maximum(x, min), max)
            def minimum(a, b):
                return -maximum(-a, -b)
                
            def clamp(x, min_val, max_val):
                return minimum(maximum(x, min_val), max_val)
                
            surr2 = clamp(ratios, Tensor(1-self.eps_clip), Tensor(1+self.eps_clip)) * advantages
            
            loss = -minimum(surr1, surr2) + 0.5 * (state_values - returns) ** 2 - 0.01 * dist_entropy
            loss = loss.mean()
            
            self.policy.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss_val += loss.data.item()
            
        self.policy_old = copy.deepcopy(self.policy)
        return total_loss_val / self.K_epochs
