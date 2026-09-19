import numpy as np

class Evaluator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def test(self, episodes=100):
        successes = 0
        for episode in range(episodes):
            state, info = self.env.reset()
            done = False
            reward = 0
            while not done:
                action = np.argmax(self.agent.q_table[state])
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            if reward == 1:
                successes += 1
        return successes / episodes