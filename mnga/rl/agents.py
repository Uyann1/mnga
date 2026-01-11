
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from IPython.display import clear_output

from mnga.autograd import Tensor, no_grad, log, mean, clamp
from mnga.nn import Module, HuberLoss
from mnga.optim import Adam
from mnga.rl.buffers import N_StepReplayBuffer, PrioritizedReplayBuffer
from mnga.rl.AgentNetworks import DQNNetwork, DuelingNetwork, CategoricalNetwork, RainbowNetwork

class DQNAgent:
    def __init__(self, env: gym.Env, memory_size: int, batch_size: int, target_update_freq: int, epsilon_decay: float, seed:int, max_epsilon: float = 1.0, min_epsilon: float = 0.01, gamma: float = 0.99):
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = N_StepReplayBuffer(obs_dim=obs_dim, size=memory_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed 
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        # network: dqn, dqn_target
        self.dqn = DQNNetwork(obs_dim, action_dim)
        self.dqn_target = DQNNetwork(obs_dim, action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = Adam(self.dqn.parameters(), lr=1e-4)

        # transition to store in memory
        self.transition = list()

        #mode: train / test
        self.is_test = False
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(Tensor(state)).argmax().data
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.clip_grad_norm(1.0)
        self.optimizer.step()

        return loss.item()
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state 
            score += reward

            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0
            
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update_freq == 0:
                    self._target_hard_update()
            

            if frame_idx % plotting_interval == 0: 
                self._plot(frame_idx, scores, losses, epsilons)
        self.env.close()

    def test(self, video_folder: str) -> None:
        self.is_test = True
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = naive_env
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]):
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"]
        done = samples["done"]

        all_q_value = self.dqn(state)
        curr_q_value = all_q_value[(np.arange(self.batch_size), action)].reshape(-1, 1)
        next_q_value = self.dqn_target(next_state).max(axis=1, keepdims=True).detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask)

        criterion = HuberLoss()

        loss = criterion(curr_q_value, target)
        
        return loss
    
    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()  


class DDQNAgent:
    def __init__(self, env: gym.Env, memory_size: int, batch_size: int, target_update_freq: int, epsilon_decay: float, seed:int, max_epsilon: float = 1.0, min_epsilon: float = 0.01, gamma: float = 0.99):
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = N_StepReplayBuffer(obs_dim=obs_dim, size=memory_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed 
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        # network: dqn, dqn_target
        self.dqn = DQNNetwork(obs_dim, action_dim)
        self.dqn_target = DQNNetwork(obs_dim, action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # loss function
        self.criterion = HuberLoss()

        # optimizer
        self.optimizer = Adam(self.dqn.parameters(), lr=1e-4)

        # transition to store in memory
        self.transition = list()

        #mode: train / test
        self.is_test = False
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(Tensor(state, requires_grad=False)).argmax().data
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm =self.optimizer.clip_grad_norm(1.0)
        self.optimizer.step()

        return loss.item(), grad_norm
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        grad_norms = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state 
            score += reward

            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0
            
            if len(self.memory) >= self.batch_size:
                loss, grad_norm = self.update_model()
                losses.append(loss)
                grad_norms.append(grad_norm)
                update_cnt += 1

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update_freq == 0:
                    self._target_hard_update()
            

            if frame_idx % plotting_interval == 0: 
                self._plot(frame_idx, scores, losses, epsilons, grad_norms)
        self.env.close()

    def test(self, video_folder: str) -> None:
        self.is_test = True
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = naive_env
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]):
        state = samples["obs"]    # batch_size x obs_dim
        next_state = samples["next_obs"]    # batch_size x obs_dim
        action = samples["acts"] # batch_size 
        reward = samples["rews"] # batch_size x 1
        done = samples["done"] # batch_size x 1 
        
        all_q_value = self.dqn(state)   # batch_size x action_dim
        curr_q_value = all_q_value[(np.arange(self.batch_size), action)].reshape(-1, 1) # batch_size x 1     <-->
        next_q_online = self.dqn(next_state).detach()   # batch_size x action_dim
        next_action = next_q_online.argmax(axis=1).data  # batch_size
        all_next_q_value = self.dqn_target(next_state).detach()  # batch_size x action_dim
        next_q_value = all_next_q_value[(np.arange(self.batch_size), next_action)].reshape(-1,1)  # batch_size x1 # <--> 

        mask = 1 - done    # batch_size x 1
        target = (reward + self.gamma * next_q_value * mask)


        loss = self.criterion(curr_q_value, target)   # scalar
        
        return loss
    
    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
        grad_norms: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 10))
        plt.subplot(221)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(222)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(223)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.subplot(224)
        plt.title('grad norms')
        plt.plot(grad_norms)
        plt.axhline(y=1.0, color='r', linestyle='--', label='clip threshold')
        plt.legend()

        plt.tight_layout()
        plt.show()  


class DDQNPERAgent:
    def __init__(
        self, 
        env: gym.Env, 
        memory_size: int, 
        batch_size: int, 
        target_update_freq: int, 
        epsilon_decay: float, 
        seed: int, 
        max_epsilon: float = 1.0, 
        min_epsilon: float = 0.05,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,          # initial beta
        beta_steps: int = 200_000   # updates to anneal beta to 1.0
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        # your existing PER buffer (unchanged)
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        self.beta_start = beta
        self.beta_end = 1.0
        self.beta_steps = beta_steps
        self.beta = beta

        self.seed = seed
        self.dqn = DQNNetwork(obs_dim, action_dim)
        self.dqn_target = DQNNetwork(obs_dim, action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.criterion = HuberLoss(delta=1.0, reduction='mean')  # default reduction for non-PER use
        self.optimizer = Adam(self.dqn.parameters(), lr=1e-4)

        self.transition = []
        self.is_test = False
        self.update_cnt = 0

    def select_action(self, state):
        if not self.is_test and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            state_t = Tensor(state, requires_grad=False)
            selected_action = self.dqn(state_t).argmax().data
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        # Optional reward clipping for stability
        reward = np.clip(reward, -1.0, 1.0)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"]
        indices = samples["indices"]

        elementwise_loss, td_error = self._compute_dqn_loss(samples)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self.optimizer.clip_grad_norm(1.0)
        self.optimizer.step()

        # update PER priorities with raw TD error
        self.memory.update_priorities(indices, td_error)

        return loss.item(), grad_norm
    
    def _compute_dqn_loss(self, samples):
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"].data.astype(int)
        reward = samples["rews"]
        done = samples["done"]

        curr_q = self.dqn(state)[np.arange(self.batch_size), action].reshape(-1, 1)

        with no_grad():
            next_action = self.dqn(next_state).argmax(axis=1).data
            next_q = self.dqn_target(next_state).data[np.arange(self.batch_size), next_action].reshape(-1, 1)
            target = reward.data + self.gamma * next_q * (1 - done.data)
            target_t = Tensor(target, requires_grad=False)

        # elementwise Huber (no reduction) for PER weighting
        elementwise_loss = HuberLoss(delta=1.0, reduction='none')(curr_q, target_t)
        # raw TD error for priorities
        td_error = np.abs((curr_q - target_t).data).reshape(-1)
        return elementwise_loss, td_error

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state, _ = self.env.reset(seed=self.seed)

        score = 0
        scores, losses, epsilons, grad_norms = [], [], [], []

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state 
            score += reward

            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0
            
            if len(self.memory) >= self.batch_size:
                loss_val, grad_norm = self.update_model()
                self.update_cnt += 1

                # beta anneal on updates
                frac = min(1.0, self.update_cnt / self.beta_steps)
                self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

                losses.append(loss_val)
                grad_norms.append(grad_norm)

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                epsilons.append(self.epsilon)

                if self.update_cnt % self.target_update_freq == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())
            
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons, grad_norms)

    def _plot(self, frame_idx, scores, losses, epsilons, grad_norms):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        clear_output(True)
        plt.figure(figsize=(20, 10))
        plt.subplot(221); plt.title(f'Frame {frame_idx}. Score: {np.mean(scores[-10:]) if scores else 0}'); plt.plot(scores)
        plt.subplot(222); plt.title('Loss'); plt.plot(losses)
        plt.subplot(223); plt.title('Epsilons'); plt.plot(epsilons)
        plt.subplot(224); plt.title('Grad Norms'); plt.plot(grad_norms); plt.axhline(y=1.0, color='r', linestyle='--', label='clip'); plt.legend()
        plt.tight_layout(); plt.show()

class DDQNPERDuelingAgent:
    def __init__(
        self, 
        env: gym.Env, 
        memory_size: int, 
        batch_size: int, 
        target_update_freq: int, 
        epsilon_decay: float, 
        seed: int, 
        max_epsilon: float = 1.0, 
        min_epsilon: float = 0.05,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,          # initial beta
        beta_steps: int = 60_000   # updates to anneal beta to 1.0
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        self.beta_start = beta
        self.beta_end = 1.0
        self.beta_steps = beta_steps
        self.beta = beta

        self.seed = seed
        self.dqn = DuelingNetwork(obs_dim, action_dim)
        self.dqn_target = DuelingNetwork(obs_dim, action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.criterion = HuberLoss(delta=1.0, reduction='mean')  # default reduction for non-PER use
        self.optimizer = Adam(self.dqn.parameters(), lr=1e-4)

        self.transition = []
        self.is_test = False
        self.update_cnt = 0

    def select_action(self, state):
        if not self.is_test and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            state_t = Tensor(np.expand_dims(state, 0), requires_grad=False)  # shape (1, obs_dim)
            selected_action = self.dqn(state_t).argmax().data
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        # Optional reward clipping for stability
        reward = np.clip(reward, -1.0, 1.0)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"]
        indices = samples["indices"]

        elementwise_loss, td_error = self._compute_dqn_loss(samples)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self.optimizer.clip_grad_norm(1.0)
        self.optimizer.step()

        # update PER priorities with raw TD error
        self.memory.update_priorities(indices, td_error)

        return loss.item(), grad_norm
    
    def _compute_dqn_loss(self, samples):
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"].data.astype(int)
        reward = samples["rews"]
        done = samples["done"]

        curr_q = self.dqn(state)[np.arange(self.batch_size), action].reshape(-1, 1)

        with no_grad():
            next_action = self.dqn(next_state).argmax(axis=1).data
            next_q = self.dqn_target(next_state).data[np.arange(self.batch_size), next_action].reshape(-1, 1)
            target = reward.data + self.gamma * next_q * (1 - done.data)
            target_t = Tensor(target, requires_grad=False)

        # elementwise Huber (no reduction) for PER weighting
        elementwise_loss = HuberLoss(delta=1.0, reduction='none')(curr_q, target_t)
        # raw TD error for priorities
        td_error = np.abs((curr_q - target_t).data).reshape(-1)
        return elementwise_loss, td_error

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state, _ = self.env.reset(seed=self.seed)

        score = 0
        scores, losses, epsilons, grad_norms = [], [], [], []

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state 
            score += reward

            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0
            
            if len(self.memory) >= self.batch_size:
                loss_val, grad_norm = self.update_model()
                self.update_cnt += 1

                # beta anneal on updates
                frac = min(1.0, self.update_cnt / self.beta_steps)
                self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

                losses.append(loss_val)
                grad_norms.append(grad_norm)

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                epsilons.append(self.epsilon)

                if self.update_cnt % self.target_update_freq == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())
            
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons, grad_norms)

    def _plot(self, frame_idx, scores, losses, epsilons, grad_norms):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        clear_output(True)
        plt.figure(figsize=(20, 10))
        plt.subplot(221); plt.title(f'Frame {frame_idx}. Score: {np.mean(scores[-10:]) if scores else 0}'); plt.plot(scores)
        plt.subplot(222); plt.title('Loss'); plt.plot(losses)
        plt.subplot(223); plt.title('Epsilons'); plt.plot(epsilons)
        plt.subplot(224); plt.title('Grad Norms'); plt.plot(grad_norms); plt.axhline(y=1.0, color='r', linestyle='--', label='clip'); plt.legend()
        plt.tight_layout(); plt.show()

class CategoricalAgent:
    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # Categorical parameters
        v_min: float = 0.0,
        v_max: float = 500.0,
        atom_size: int = 51,
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        # Using your standard ReplayBuffer
        self.memory = N_StepReplayBuffer(obs_dim=obs_dim, size=memory_size, batch_size=batch_size)
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        # support = [v_min, ..., v_max]
        self.support = Tensor.linspace(self.v_min, self.v_max, self.atom_size)

        # Networks using your framework's Module system
        self.dqn = CategoricalNetwork(obs_dim, action_dim, atom_size, self.support)
        self.dqn_target = CategoricalNetwork(obs_dim, action_dim, atom_size, self.support)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.optimizer = Adam(self.dqn.parameters(), lr=1e-4)

        self.transition = list()
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action using epsilon-greedy policy."""
        if not self.is_test and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            # Forward pass returns Expected Value Q(s,a)
            state_t = Tensor(state, requires_grad=False)
            selected_action = self.dqn(state_t).argmax().data
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> float:
        """Update the model using Categorical Loss."""
        samples = self.memory.sample_batch()

        # In your framework, loss calculation usually happens in _compute_dqn_loss
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        # Adding gradient clipping for stability in distributional RL
        self.optimizer.clip_grad_norm(10.0)
        self.optimizer.step()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, Tensor]) -> Tensor:
        """Categorical DQN (C51) Loss implementation."""
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"]
        done = samples["done"]
        
        # Distance between atoms
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with no_grad():
            # 1. Selection: Online network chooses best action for next_state
            next_action = self.dqn(next_state).argmax(axis=1)
            
            # 2. Evaluation: Target network gets distribution for next_state
            next_dist = self.dqn_target.dist(next_state)
            # Pick distribution for the chosen next_action
            next_dist = next_dist[np.arange(self.batch_size), next_action.data]

            # 3. Project next_dist onto current support (Tz)
            # Tz = R + gamma * z
            t_z = reward + (Tensor(1.0) - done) * self.gamma * self.support
            t_z = t_z.clamp(min_value=self.v_min, max_value=self.v_max)
            
            # 4. Compute indices for projection
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Fix indices to be within [0, atom_size - 1]
            l = l.clamp(min_value=0, max_value=self.atom_size - 1)
            u = u.clamp(min_value=0, max_value=self.atom_size - 1)

            # 5. Distribute probability mass (m_l and m_u)
            offset = Tensor.linspace(
                0, (self.batch_size - 1) * self.atom_size, self.batch_size
            ).long().view(-1, 1)

            proj_dist = Tensor.zeros((self.batch_size, self.atom_size), requires_grad=False)
            
            # Use index_add_ style logic adapted to your framework
            # This implements the projection: m_l ← m_l + p(s', a') * (u - b)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # 6. Current predicted distribution
        dist = self.dqn.dist(state)
        action_indices = action.data.flatten().astype(int)
        log_p = log(dist[np.arange(self.batch_size), action_indices].clamp(min_value=1e-5))

        # 7. Cross-Entropy Loss: - Σ (proj_dist * log_p)
        loss = -(proj_dist * log_p).sum(axis=1).mean()

        return loss

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state, _ = self.env.reset(seed=self.seed)
        
        update_cnt = 0
        scores, losses, epsilons = [], [], []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss_val = self.update_model()
                losses.append(loss_val)
                update_cnt += 1
                
                # Epsilon decay logic
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                epsilons.append(self.epsilon)
                
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, frame_idx, scores, losses, epsilons):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131); plt.title(f'Frame {frame_idx}. Score: {np.mean(scores[-10:]) if scores else 0}'); plt.plot(scores)
        plt.subplot(132); plt.title('Loss'); plt.plot(losses)
        plt.subplot(133); plt.title('Epsilons'); plt.plot(epsilons)
        plt.show()

class RainbowAgent:
    def __init__(
        self, 
        env: gym.Env, 
        memory_size: int, 
        batch_size: int, 
        target_update: int, 
        seed: int, 
        gamma: float = 0.99,
        alpha: float = 0.2, 
        beta: float = 0.6,
        v_min: float = 0.0, 
        v_max: float = 500.0, 
        atom_size: int = 51,
        n_step: int = 3
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.n_step = n_step

        # PER
        self.beta = beta
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma)

        # memory for N-step learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = N_StepReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)
            
        # Categorical DQN (Distributional RL) parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = Tensor.linspace(self.v_min, self.v_max, atom_size)

        # Networks: dqn, dqn_target
        self.dqn = RainbowNetwork(obs_dim, action_dim, self.atom_size, self.support)
        self.dqn_target = RainbowNetwork(obs_dim, action_dim, self.atom_size, self.support)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        # Note: In custom frameworks, ensuring eval mode depends on your Module implementation
        
        # Optimizer
        self.optimizer = Adam(self.dqn.parameters())

        # Transition to store in memory
        self.transition = list()
        # Mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        selected_action = self.dqn(Tensor(state, requires_grad=False)).argmax().data
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            # 1. Store the raw transition in the primary (1-step) buffer
            self.memory.store(*self.transition)
            
            # 2. Store the same raw transition in the N-step buffer
            if self.use_n_step:
                self.memory_n.store(*self.transition)
        
        return next_state, reward, done

    def update_model(self) -> float:
        # 1. PER: Sample a batch and get Importance Sampling weights
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"]
        indices = samples["indices"]
        
        self.dqn.reset_noise()
        # 2. 1-step Learning loss calculation
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: Weighted mean of the 1-step loss
        loss = mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma_n = self.gamma ** self.n_step
            # Sample N-step transitions using the EXACT same indices as the 1-step batch
            samples_n = self.memory_n.sample_batch_from_indices(indices)
            # Compute loss for the N-step transitions
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma_n)
            # Combine losses (1-step + N-step) to reduce variance
            elementwise_loss = elementwise_loss + elementwise_loss_n
            # Re-calculate the final weighted loss
            loss = mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.clip_grad_norm(10.0)
        self.optimizer.step()

        # PER: Update priorities in the SumTree
        loss_for_prior = elementwise_loss.data
        self.memory.update_priorities(indices, loss_for_prior)
        
        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 100):
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if target update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
            
            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)
                
        self.env.close()

    def test(self, video_folder: str) -> None:
        self.is_test = True
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, Tensor], gamma: float) -> Tensor:
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"]
        done = samples["done"]
        
        # Constants
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with no_grad():
            # A) Selection: Online network chooses the best action for next_state
            # Note: We must reset noise for consistent evaluation if using NoisyNets
            # self.dqn.reset_noise() # Optional: usually we just use the current noise state
            next_action = self.dqn(next_state).argmax(axis=1) 
            
            # B) Evaluation: Target network calculates the distribution for next_state
            next_dist = self.dqn_target.dist(next_state) 
            
            # C) Select the distribution corresponding to the best action
            # Shape: (batch_size, atom_size)
            next_dist = next_dist[np.arange(self.batch_size), next_action.data]

            # ---------------------------------------------------------
            # 3. N-STEP RETURN CALCULATION (Distribution Shift)
            # ---------------------------------------------------------
            # T_z = R_n + (gamma^n) * z
            #gamma_n = self.gamma ** self.n_step
            
            # rewards and dones are (batch, 1), support is (atom_size)
            # We broadcast to (batch, atom_size)
            t_z = reward + (Tensor(1.0) - done) * gamma * self.support
            t_z = t_z.clamp(min_value=self.v_min, max_value=self.v_max)
            
            # ---------------------------------------------------------
            # 4. PROJECTION (The "C51" Algorithm)
            # ---------------------------------------------------------
            # Map the continuous t_z values back to the discrete support indices [0, atom_size-1]
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Ensure indices are within bounds [0, atom_size - 1]
            l = l.clamp(min_value=0, max_value=self.atom_size - 1)
            u = u.clamp(min_value=0, max_value=self.atom_size - 1)

            # Distribute probability mass
            # We need to flatten the batch to use index_add_ (or scatter_add) efficiently
            offset = Tensor.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long().view(-1, 1)
            
            # Projected distribution (m in the paper)
            proj_dist = Tensor.zeros((self.batch_size, self.atom_size), requires_grad=False)
            
            # Simple Python loop implementation (easier to debug, slightly slower)
            # Or use the vectorized index_add if your Tensor lib supports it
            
            # Vectorized Logic (assuming PyTorch-like index_add_):
            # We add mass to the lower index (l) based on distance from upper (u)
            final_l_indices = (l + offset).view(-1)
            final_u_indices = (u + offset).view(-1)
            
            proj_dist.view(-1).index_add_(0, final_l_indices, (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, final_u_indices, (next_dist * (b - l.float())).view(-1))
        

        # 5. Current Distribution (Requires Gradients)
        dist = self.dqn.dist(state)
        
        # Extract the probability of the action actually taken
        # Your GetItem function handles this via dist[...]
        action_indices = action.data.flatten().astype(int)
        current_dist = dist[np.arange(self.batch_size), action_indices]
        
        # Stable Log: log(p) clamped to avoid -inf
        log_p = log(current_dist.clamp(min_value=1e-5))
        
        # Categorical Cross-Entropy Loss: - Σ (target_dist * log(pred_dist))
        elementwise_loss = -(proj_dist * log_p).sum(axis=1)

        return elementwise_loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, frame_idx: int, scores: List[float], losses: List[float]):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:]) if scores else 0))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()