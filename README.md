# Neuro RL Game Theory

Reinforcement Learning pipeline implementing Q-Learning on Gymnasium FrozenLake-v1 environment accompanied by a Game Theory and Reinforcement Learning analysis of a multi agent system.

## Project Overview

This repository contains two main components:
1. Practical RL Implementation: A Q-Learning agent built from scratch using Gymnasium on a 4x4 discrete grid world (FrozenLake-v1).
2. Multi Agent Case Study Analysis: A theoretical and structural evaluation mapping Router, Retriever, Generator, and Critic agents to RL and non-zero-sum cooperative Game Theory
3. frameworks.

## Results and Performance

- Training Duration: 2000 episodes with an epsilon greedy exploration strategy
- Training Success Rate: Achieved approximately 100% rolling success rate over the final 100 episodes
- Test Evaluation: 100% success rate across 100 evaluation episodes with exploration turned off (epsilon = 0)

## Multi Agent Game Theory Mapping

| Actor / Agent | State | Action | Tool | Reward / Payoff | Next State / Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Router Agent | New query received | Classify query path | Groq LLM | Correct classification | Route to correct retriever |
| Retriever Agent | Topic set | Search vector store | ChromaDB | Relevant chunks retrieved | Context added to state |
| Generator Agent | Context retrieved | Generate response | Groq LLM | Coherent answer | Answer added to state |
| Critic Agent | Answer generated | Approve / Reject | Groq LLM | Positive reward / Retry penalty | Complete process or retry loop |

## Setup and Execution

### 1. Installation
git clone https://github.com/manaal007/neuro-rl-game-theory.git
cd neuro-rl-game-theory
pip install -r requirements.txt

### 2. Run Training and Evaluation
python src/main.py

### 3. Run Unit Tests
pytest tests/