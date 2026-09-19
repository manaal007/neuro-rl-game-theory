import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from q_agent import QLearningAgent

def test_agent_initializes_zero_q_table():
    agent = QLearningAgent(n_states=16, n_actions=4)
    assert agent.q_table.shape == (16, 4)
    assert np.all(agent.q_table == 0)

def test_agent_updates_q_value():
    agent = QLearningAgent(n_states=16, n_actions=4)
    agent.update(state=0, action=1, reward=1, new_state=1)
    assert agent.q_table[0, 1] != 0