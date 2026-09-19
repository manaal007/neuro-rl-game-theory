import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.8, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.001, min_epsilon=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, env):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, new_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[new_state])
        self.q_table[state, action] = old_value + self.lr * (reward + self.gamma * next_max - old_value)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)