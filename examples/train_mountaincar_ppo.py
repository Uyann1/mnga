import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnga.rl import PPOAgent
from mnga.utils import plot_learning_curve

def train():
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, 'videos/mountaincar_ppo', episode_trigger=lambda x: x % 100 == 0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # PPO might need more epochs or smaller learning rate for stability
    # Increase LR and epochs for faster learning
    agent = PPOAgent(state_dim, action_dim, lr=0.005, gamma=0.99, K_epochs=20, eps_clip=0.2)
    
    episodes = 1000
    rewards_history = []
    loss_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Reward Shaping: Incentivize height and velocity
            # State: [position, velocity]
            # Position: [-1.2, 0.6], Goal: 0.5
            # Velocity: [-0.07, 0.07]
            
            pos = next_state[0]
            vel = next_state[1]
            
            # Add potential energy reward (height)
            # Height is roughly proportional to sin(3 * pos)
            # But simpler: closer to 0.5 is better.
            
            # Standard reward is -1.
            # Let's add a term for height.
            # Height ~ pos.
            
            modified_reward = reward + 10 * abs(vel) # Encourage moving
            
            # If reached goal (done and not truncated), give big bonus
            if done and not truncated:
                modified_reward += 100.0
                print(f"Goal Reached! Episode {episode}")
                
            loss = agent.step(state, action, modified_reward, next_state, done or truncated, log_prob, value)
            if loss is not None:
                loss_history.append(loss)
            
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
            
    plot_learning_curve(rewards_history, loss_history, filename='mountaincar_ppo.png')

if __name__ == "__main__":
    train()
