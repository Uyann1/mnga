import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from .segment_tree import SumSegmentTree, MinSegmentTree
from ..autograd import Tensor

class N_StepReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int=32, n_step: int=1, gamma: float=0.99):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

        # for N-step learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        
        # For Sampling
        self.obs_tensor = Tensor.zeros((self.batch_size, obs_dim), requires_grad=False)
        self.next_obs_tensor = Tensor.zeros((self.batch_size, obs_dim), requires_grad=False)
        self.acts_tensor = Tensor.zeros((self.batch_size,), dtype=np.int64, requires_grad=False)
        self.rews_tensor = Tensor.zeros((self.batch_size, 1), requires_grad=False)
        self.done_tensor = Tensor.zeros((self.batch_size, 1), requires_grad=False)
    
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)
        
        stored_indices = []

        if done:
            # When the episode ends, we must process everything remaining in the deque
            while len(self.n_step_buffer) > 0:
                rew, next_obs, done = self._get_n_step_info()
                obs, act = self.n_step_buffer[0][:2]
                stored_indices.append(self._push_to_main(obs, act, rew, next_obs, done))
                self.n_step_buffer.popleft()
            return stored_indices
        
        if len(self.n_step_buffer) == self.n_step:
            rew, next_obs, done = self._get_n_step_info()
            obs, act = self.n_step_buffer[0][:2]
            stored_indices.append(self._push_to_main(obs, act, rew, next_obs, done))
        return stored_indices

    
    def sample_batch(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        self.obs_tensor.data[:] = self.obs_buf[indices]
        self.next_obs_tensor.data[:] = self.next_obs_buf[indices]
        self.acts_tensor.data[:] = self.acts_buf[indices]
        self.rews_tensor.data[:] = self.rews_buf[indices].reshape(-1, 1)
        self.done_tensor.data[:] = self.done_buf[indices].reshape(-1, 1)
        return dict(
            obs=self.obs_tensor,
            next_obs=self.next_obs_tensor,
            acts=self.acts_tensor,
            rews=self.rews_tensor,
            done=self.done_tensor,
            indices=indices,
        )
    
    def sample_batch_from_indices(self, indices: np.ndarray):
        self.obs_tensor.data[:] = self.obs_buf[indices]
        self.next_obs_tensor.data[:] = self.next_obs_buf[indices]
        self.acts_tensor.data[:] = self.acts_buf[indices]
        self.rews_tensor.data[:] = self.rews_buf[indices].reshape(-1, 1)
        self.done_tensor.data[:] = self.done_buf[indices].reshape(-1, 1)
        return dict(
            obs=self.obs_tensor,
            next_obs=self.next_obs_tensor,
            acts=self.acts_tensor,
            rews=self.rews_tensor,
            done=self.done_tensor,
        )
    
    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        rew, next_obs, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            #Discounterd reward: r + gamma * future_reward
            rew = r + self.gamma * rew * (1 - d)
            # Check if done occurred in the n-step transitions
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return rew, next_obs, done
    
    def _push_to_main(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        idx = self.ptr
        
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.done_buf[idx] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return idx
    
    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(N_StepReplayBuffer):
    def __init__(self, obs_dim: int, size: int, batch_size: int=32, alpha: float=0.6, n_step: int=1, gamma: float=0.99, eps: float=1e-6):
        super().__init__(obs_dim, size, batch_size, n_step, gamma)
        self.alpha = alpha
        self.max_priority = 1.0
        self.eps = eps

        # Segment Tree capacity must be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < self.max_size: 
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.weights_tensor = Tensor.zeros((self.batch_size, 1), requires_grad=False)
    
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        indices = super().store(obs, act, rew, next_obs, done)
        priority = self.max_priority ** self.alpha
        for idx in indices:
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
               
    
    def sample_batch(self, beta: float=0.4):

        indices = self._stratified_sampling()

        # Vectorized IS weight calculation 
        p_total = self.sum_tree.total
        # Get priorities for all samples indices at one
        p_samples = np.array([self.sum_tree[idx] for idx in indices]) 

        # Weights formula: (N * P(i)) ^ -beta
        probs = p_samples / p_total
        weights = (self.size * probs) ** (-beta)

        # Normalize by max possible weight to keep updates stable
        p_min = self.min_tree.total / p_total
        max_weight = (self.size * p_min) ** (-beta)
        weights /= max_weight
        self.weights_tensor.data[:] = weights.reshape(-1, 1)

        return {
            **super().sample_batch_from_indices(indices),
            "weights": self.weights_tensor,
            "indices": indices,
        }
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):

        for idx, priority in zip(indices, priorities):
            priority_alpha = (priority + self.eps) ** self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha
            self.max_priority = max(self.max_priority, priority)

    def _stratified_sampling(self) -> List[int]:
        #Proportional sampling via segment tree 
        indices = []
        segment = self.sum_tree.total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            # Boundary safety check
            indices.append(min(idx, self.size - 1))
        return indices  