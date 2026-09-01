"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # builds a 2D float64 numpy array of zeros sized by states and actions.
    
    return np.zeros((num_states, num_actions))

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # indexes the row for `state` and return its maximum value
    return max(q_table[state])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # returns argmax over the action axis for this state's Q values
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
import numpy as np

def sample_random_action(action_space):
    # draws a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Returns True with probability epsilon"""
    # draws a uniform sample from rng and compare it to epsilon
    if rng.random() < epsilon:
        return True
    return False

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # with prob epsilon explore via action_space, else take greedy action
    if should_explore(epsilon, rng) == True:
        return sample_random_action(action_space)
    return greedy_action(q_table, state)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # rises first and then decays later
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # computes bootstrap target, zeroing the bootstrap when done.
    # conceptually, what I got now + what I expect to get later
    if done:
        return reward
    return reward + gamma * max_q_value(q_table, next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # return temporal diff error: target minus current Q(state, action)
    return target - q_table[state][action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    """
    atomic learning operation of tabular Q-learning. Q(s,a) += lr * (target - Q(s,a))
    """
    # apply in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    q_table[state][action] += alpha * td_error(target, q_table, state, action)
    return q_table[state][action]

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    """
    selects epsilon-greedy action, step env, apply Q-learning update, 
    return (next_state, reward, done)
    """
    # picks action with epsilon-greedy
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)

    # steps environment
    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    # updates q-table
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)

    return (int(next_state), float(reward), done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # resets env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    state, info = env.reset()
    steps = 0
    while steps < 200:
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        if done:
            break
        steps += 1

    return reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    # trains a Q-learning agent for num_episodes; return (q_table, returns)

    # rng/seeding setup
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.reset(seed=seed)

    # q-table initialization
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = init_q_table(num_states, num_actions)

    # initialize returns/naming
    episode_returns = []
    epsilon = epsilon_start

    # run num_episodes training episodes
    for _ in range(num_episodes):
        reward = run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps)
        episode_returns.append(float(reward))
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)

    return q_table, episode_returns

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

