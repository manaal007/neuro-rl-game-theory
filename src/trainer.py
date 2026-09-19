class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self, episodes=2000):
        rewards_per_episode = []
        for episode in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.choose_action(state, self.env)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.agent.update(state, action, reward, new_state)
                state = new_state
                total_reward += reward
            rewards_per_episode.append(total_reward)
            self.agent.decay_epsilon()
        return rewards_per_episode