import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards, losses, filename='training_plot.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Episode Reward')
    # Calculate moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
