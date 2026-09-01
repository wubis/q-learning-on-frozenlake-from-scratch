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

# Step 7 - decay_epsilon (not yet solved)
# TODO: implement

# Step 8 - td_target (not yet solved)
# TODO: implement

# Step 9 - td_error (not yet solved)
# TODO: implement

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

