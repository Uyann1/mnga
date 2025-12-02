import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Handle state_dim being a tuple for images or int for vectors
        if isinstance(state_dim, int):
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        else:
            self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
            
        self.actions = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )

    def __len__(self):
        return self.size
