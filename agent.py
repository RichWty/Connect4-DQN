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

    def __init__(self,name, hyperparameters_set, checkpoint_dir):
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
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, rewards, new_states, next_valid_masks, dones = zip(*mini_batch)
        
        states = torch.stack(states).unsqueeze(1)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states).unsqueeze(1)
        rewards = torch.stack(rewards)
        next_valid_masks = torch.stack(next_valid_masks)
        dones = torch.stack(dones)

        with torch.no_grad():
            # Double DQN avec masquage des coups illégaux
            policy_next_q = policy_dqn(new_states)
            policy_next_q[~next_valid_masks] = float('-inf')  # On ignore les colonnes pleines
            best_action = policy_next_q.argmax(dim=1)
            
            target_next_q = target_dqn(new_states)
            
            # ATTENTION AU SIGNE '-' : Puisque le jeu est à somme nulle et que new_states 
            # est vu du point de vue de l'adversaire (inversé dans train.py), la valeur 
            # pour nous est l'opposée de la valeur pour l'adversaire.
            target_q = rewards - (1 - dones) * self.discount_factor_g * target_next_q.gather(1, best_action.unsqueeze(1)).squeeze(1)

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())


class RandomAgent:
    def __init__(self, name):
        self.name = name

    def select_action(self, state, vali_actions, is_training):
        return choice(vali_actions)

        