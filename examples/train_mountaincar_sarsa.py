import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnga.rl import SARSAAgent
from mnga.utils import plot_learning_curve

def train():
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, 'videos/mountaincar_sarsa', episode_trigger=lambda x: x % 100 == 0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # SARSA often needs more exploration or better features for MountainCar, but let's try with MLP
    agent = SARSAAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    
    episodes = 1000 # MountainCar is hard
    rewards_history = []
    loss_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # For SARSA, we need to be careful about the first action
            # The agent.act handles next_action storage
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            loss = agent.step(state, action, reward, next_state, done or truncated)
            if loss is not None:
                loss_history.append(loss)
            
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")
            
    plot_learning_curve(rewards_history, loss_history, filename='mountaincar_sarsa.png')

if __name__ == "__main__":
    train()
