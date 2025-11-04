# Taxi Q-Learning Agent

A reinforcement learning implementation using Q-Learning to solve the OpenAI Gym Taxi-v3 environment.

## Overview

This project implements a Q-Learning algorithm to train an agent that learns to pick up and drop off passengers in the Taxi environment. The agent learns optimal policies through trial and error, gradually improving its performance over thousands of episodes.

## Environment

**Taxi-v3** is a classic reinforcement learning environment where:
- A taxi must pick up a passenger at one location and drop them off at another
- The environment has 500 discrete states and 6 possible actions
- Actions include: move south, north, east, west, pickup, and dropoff
- The agent receives rewards for successful delivery and penalties for illegal moves

## Features

- **Q-Learning Implementation**: Classical tabular Q-Learning with epsilon-greedy exploration
- **Action Masking**: Uses valid action masks to prevent illegal moves during exploration
- **Configurable Hyperparameters**: Two parameter sets provided for experimentation
- **Training Visualization**: Displays average rewards per thousand episodes
- **Trained Agent Demo**: Visualizes the trained agent's performance

## Requirements

```bash
pip install gym numpy ipython
```

## Usage

### Training the Agent

Run the Jupyter notebook cells sequentially to:
1. Initialize the environment and Q-table
2. Configure hyperparameters
3. Train the agent for 10,000 episodes
4. View training progress

### Hyperparameters (Set 1)

- **Episodes**: 10,000
- **Maximum Steps**: 1,000 per episode
- **Learning Rate (α)**: 0.1
- **Discount Rate (γ)**: 0.99
- **Exploration Rate (ε)**: Decays from 1.0 to 0.01
- **Exploration Decay**: 0.001

### Alternative Configuration (Set 2)

An optimized parameter set is included in comments, featuring:
- Automatically calculated maximum steps based on discount rate
- Adaptive learning rate: `1000 / episodes`
- Optimal exploration decay: `-log(0.00001) / episodes`

## Training Results

The agent shows significant improvement over time:
- **Episodes 1-1000**: Average reward -148.49 (random exploration)
- **Episodes 2000-3000**: Rapid improvement as exploitation increases
- **Episodes 4000-10000**: Stabilizes around 7.5-7.8 average reward

## Algorithm

The Q-Learning update rule:

```
Q(s,a) ← (1-α)Q(s,a) + α[r + γ·max Q(s',a')]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `α` = learning rate
- `γ` = discount factor

## Viewing the Trained Agent

The final cell demonstrates the trained agent's performance by running 5 episodes with rendering enabled, allowing you to watch the taxi navigate optimally.

## Project Structure

```
├── taxi_qlearning.ipynb   # Main implementation notebook
└── README.md              # This file
```

## Future Improvements

- Implement Deep Q-Learning (DQN) for comparison
- Add performance metrics visualization (success rate, steps per episode)
- Experiment with different exploration strategies (Boltzmann, UCB)
- Save and load trained Q-tables
- Compare performance across different hyperparameter configurations

## License

This project is open source and available under the MIT License.

## References

- [OpenAI Gym Taxi-v3 Documentation](https://gymnasium.farama.org/environments/toy_text/taxi/)
- Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD thesis, Cambridge University.
