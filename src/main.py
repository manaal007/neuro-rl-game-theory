import sys
from pathlib import Path

# Add the directory containing main.py to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from q_agent import QLearningAgent
from trainer import Trainer
from evaluator import Evaluator

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from q_agent import QLearningAgent
from trainer import Trainer
from evaluator import Evaluator

def run():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n)

    trainer = Trainer(agent, env)
    rewards = trainer.train(episodes=2000)
    print("Training success rate (last 100):", np.mean(rewards[-100:]))

    evaluator = Evaluator(agent, env)
    test_result = evaluator.test(episodes=100)
    print("Test success rate:", test_result)

    window = 50
    rolling_avg = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
    plt.plot(rolling_avg)
    plt.xlabel('Episode')
    plt.ylabel('Success rate (rolling average)')
    plt.title('Q-Learning Training Progress on FrozenLake')
    plt.savefig('../outputs/training_progress.png')

    return agent, test_result

if __name__ == '__main__':
    run()