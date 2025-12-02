import gymnasium as gym
import numpy as np
import sys
import os
import cv2

# Add the project root to the path so we can import mnga
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnga.rl import DQNAgent
from mnga.nn import Module, Conv2d, Linear, ReLU
from mnga.autograd import Tensor
from mnga.utils import plot_learning_curve

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def reset(self):
        obs, info = self.env.reset()
        return self._process(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process(obs), reward, done, truncated, info

    def _process(self, obs):
        # Grayscale
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension (C, H, W)
        obs = np.expand_dims(obs, axis=0)
        return obs

class ConvQNetwork(Module):
    def __init__(self, in_channels, action_dim):
        super().__init__()
        # Handle if in_channels is passed as tuple (C, H, W)
        if isinstance(in_channels, tuple):
            in_channels = in_channels[0]
            
        # Input: (N, C, 84, 84)
        self.conv1 = Conv2d(in_channels, 16, kernel_size=8, stride=4) # -> (16, 20, 20)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2) # -> (32, 9, 9)
        self.relu2 = ReLU()
        
        # Flatten size: 32 * 9 * 9 = 2592
        self.fc1 = Linear(32 * 9 * 9, 256)
        self.relu3 = ReLU()
        self.fc2 = Linear(256, action_dim)
        
        self._parameters['conv1'] = self.conv1
        self._parameters['conv2'] = self.conv2
        self._parameters['fc1'] = self.fc1
        self._parameters['fc2'] = self.fc2

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Flatten
        N = x.shape[0]
        x = x.reshape(N, -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train():
    # Try to make Pong, if not available, warn user
    # Debug: Print available environments
    import ale_py
    print("Registered environments:", [env_id for env_id in gym.envs.registry.keys() if 'Pong' in env_id])

    try:
        env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
    except Exception as e:
        print(f"Could not make PongNoFrameskip-v4: {e}")
        try:
            env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
        except Exception as e2:
            print(f"Could not make ALE/Pong-v5: {e2}")
            print("Please install atari dependencies: pip install gym[atari] gym[accept-rom-license]")
            return

    env = AtariPreprocessing(env)
    
    # We need to override the agent's Q-network with our ConvQNetwork
    # Since DQNAgent creates a simple MLP by default, we'll subclass or just monkey-patch it
    # For cleanliness, let's subclass
    
    class AtariDQNAgent(DQNAgent):
        def __init__(self, *args, **kwargs):
            # Pass ConvQNetwork as the network class
            kwargs['network_cls'] = ConvQNetwork
            super().__init__(*args, **kwargs)
            # No need to manually replace networks anymore if we pass the class
            
    action_dim = env.action_space.n
    # Use a smaller buffer size to save memory and time for this demo
    agent = AtariDQNAgent(state_dim=(1, 84, 84), action_dim=action_dim, lr=1e-4, buffer_size=1000, batch_size=32)
    
    episodes = 5 # Run very few episodes as it will be slow
    rewards_history = []
    loss_history = []
    
    print("Starting training on Pong-v4...")
    print("Note: This will be SLOW due to pure Python/NumPy convolution implementation.")
    
    for episode in range(episodes):
        state, _ = env.reset()
        # Normalize state
        state = state.astype(np.float32) / 255.0
        
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Normalize next_state
            next_state = next_state.astype(np.float32) / 255.0
            
            loss = agent.step(state, action, reward, next_state, done or truncated)
            if loss is not None:
                loss_history.append(loss)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count}", end='\r')
            
            # Limit steps per episode for demo purposes if it takes too long
            if step_count > 200:
                truncated = True
            
        rewards_history.append(episode_reward)
        
        agent.update_target_network()
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Steps: {step_count}")
            
    plot_learning_curve(rewards_history, loss_history, filename='pong_dqn.png')
    print("Training finished. Plot saved to pong_dqn.png")

if __name__ == "__main__":
    train()
