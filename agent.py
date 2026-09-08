from random import choice
import torch
import yaml
import random
import os

from dqn import DQN
from Minimax import IA_Minimax

RUNS_DIR = 'runs'
CHECKPOINTS_DIR = 'checkpoints'
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    """Deep Q-Learning Agent managing policy/target networks, exploration, and training.
    
    Attributes:
        name (str): Identifier name for the agent (e.g., 'Joueur 1').
        hyperparameters_set (str): Key matching the configuration set in hyperparameters.yml.
        replay_memory_size (int): Max capacity of the replay memory.
        batch_size (int): Minibatch size sampled for training optimization.
        epsilon (float): Current probability of choosing a random action (exploration).
        epsilon_decay (float): Multiplicative decay applied to epsilon after each episode.
        epsilon_min (float): Floor value for epsilon exploration.
        learning_rate (float): Optimizer step size.
        discount_factor_g (float): Bellman discount factor gamma.
        network_update_freq (int): Frequency (in steps) to update target network weights.
        policy_dqn (DQN): Active Q-network being optimized.
        target_dqn (DQN): Delayed target Q-network stabilizing Bellman targets.
        loss_fn (nn.Module): Loss function (MSE).
        optimizer (torch.optim.Optimizer): Optimization algorithm (Adam).
    """

    def __init__(self, name, hyperparameters_set, checkpoint_dir):
        """Initialize the Agent, loading hyperparameters and network weights.
        
        Args:
            name (str): Agent identifier name.
            hyperparameters_set (str): Hyperparameter group name from hyperparameters.yml.
            checkpoint_dir (str or None): Directory containing a checkpoint to load, if any.
        """
        with open('hyperparameters.yml', 'r') as f:
            all_hyperparameters = yaml.safe_load(f)
            hyperparameters = all_hyperparameters[hyperparameters_set]
        
        self.hyperparameters_set = hyperparameters_set
        self.name = name

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_start = hyperparameters['epsilon_start']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_update_freq = hyperparameters['network_update_freq']

        self.policy_dqn = DQN(state_dim=42, action_dim=7).to(device)
        if checkpoint_dir:
            self.policy_dqn.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{self.hyperparameters_set}.pth')))
        self.target_dqn = DQN(state_dim=42, action_dim=7).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

        self.epsilon = self.epsilon_start

        self.MODEL_FILE = os.path.join(CHECKPOINTS_DIR, f'{self.hyperparameters_set}-{name}.pth')
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}-{name}.log')

        self.loss = 0

    def select_action(self, state, vali_actions, is_training=True):
        """Select an action using epsilon-greedy exploration with legal action masking.
        
        Args:
            state (torch.Tensor): Board tensor oriented to the current player's perspective.
            vali_actions (list[int]): Indices of columns that are currently playable.
            is_training (bool): If True, applies epsilon-greedy exploration; greedy otherwise.
            
        Returns:
            int: Selected column index (0-6).
        """
        if is_training and random.random() < self.epsilon:
            return choice(vali_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_dqn(state.unsqueeze(0).unsqueeze(0)).squeeze()
                masked_q_values = torch.full_like(q_values, float('-inf'))
                for a in vali_actions:
                    masked_q_values[a] = q_values[a]
                return masked_q_values.argmax().item()

    def update_epsilon(self):
        """Decay the exploration rate epsilon by epsilon_decay, bounded below by epsilon_min."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """Perform one optimization step using Double DQN and zero-sum Bellman updates.
        
        Args:
            mini_batch (list[tuple]): Sampled transitions containing:
                (states, actions, rewards, new_states, next_valid_masks, dones).
            policy_dqn (DQN): Network evaluating action selection and policy Q-values.
            target_dqn (DQN): Target network computing target state-values.
        """
        states, actions, rewards, new_states, next_valid_masks, dones = zip(*mini_batch)
        
        states = torch.stack(states).unsqueeze(1)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states).unsqueeze(1)
        rewards = torch.stack(rewards)
        next_valid_masks = torch.stack(next_valid_masks)
        dones = torch.stack(dones)

        with torch.no_grad():
            # Double DQN with illegal action masking
            policy_next_q = policy_dqn(new_states)
            policy_next_q[~next_valid_masks] = float('-inf')  # Ignore full columns
            best_action = policy_next_q.argmax(dim=1)
            
            target_next_q = target_dqn(new_states)
            
            # Zero-sum property: new_states is seen from the opponent's perspective,
            # so our value is the negative of the opponent's best expected return.
            target_q = rewards - (1 - dones) * self.discount_factor_g * target_next_q.gather(1, best_action.unsqueeze(1)).squeeze(1)

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss

    def update_target_network(self):
        """Copy weights from the policy network to the target network."""
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())


class RandomAgent:
    """Baseline agent selecting uniform random moves among valid options.
    
    Attributes:
        name (str): Identifier name for the random agent.
    """

    def __init__(self, name):
        """Initialize the random agent.
        
        Args:
            name (str): Agent identifier name.
        """
        self.name = name

    def select_action(self, state, vali_actions, is_training):
        """Uniformly pick a random action from the list of valid column indices.
        
        Args:
            state (torch.Tensor or np.ndarray): Current board state (unused).
            vali_actions (list[int]): Playable column indices.
            is_training (bool): Training flag (unused).
            
        Returns:
            int: Randomly chosen column index.
        """
        return choice(vali_actions)