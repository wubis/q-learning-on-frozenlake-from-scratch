# Q-Learning on FrozenLake from Scratch

This project implements tabular Q-learning from scratch on Gymnasium's FrozenLake environment.

The environment is a small grid world. At each step, the agent observes its current state, chooses an action, transitions to a new state, and receives a reward. The Q-learning algorithm gradually estimates which actions are most valuable in each state.

## How it works

The agent stores its estimates in a Q-table:

```text
Q[state, action]
```

Each entry represents the estimated long-term value of taking a particular action from a particular state.

At the beginning of training, the table contains little useful information, so the agent needs to explore. Action selection uses an epsilon-greedy policy:

```text
with probability ε → choose a random action
otherwise          → choose the action with the highest Q-value
```

As training progresses, `ε` decreases, so behavior gradually shifts from exploration toward exploitation of the learned values.

After each environment transition, the selected Q-value is updated using:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

Here:

* `α` controls how strongly new information changes the current estimate
* `γ` controls how much future reward contributes to the estimate
* `r` is the immediate reward
* `max Q(s', a')` is the best estimated value available from the next state

The quantity

```text
r + γ max Q(s',a')
```

is the TD target, and the difference between that target and the current Q-value is the TD error.

Repeated updates propagate information about successful transitions backward through the state space until useful action values emerge.

## Training flow

The implementation is broken into small pieces that correspond directly to the Q-learning process:

```text
initialize Q-table
        ↓
select actions with epsilon-greedy
        ↓
step through the environment
        ↓
compute TD target and TD error
        ↓
update Q-values
        ↓
repeat across episodes
        ↓
extract the greedy policy
        ↓
evaluate the learned behavior
```

The project includes functions for:

* Q-table initialization and lookup
* greedy and random action selection
* epsilon-greedy exploration
* epsilon decay
* TD target and TD error calculation
* Q-learning updates
* environment interaction
* training episodes
* greedy policy extraction
* final policy evaluation

## Running the project

Install the dependencies:

```bash
pip install numpy gymnasium
```

Then run:

```bash
python scaffold.py
```

The script trains the Q-table over multiple episodes, extracts the greedy policy produced by the learned values, and evaluates how often that policy reaches the goal.
