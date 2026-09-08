import os
import sys
import yaml
import torch
import time
from random import choice

import Puissance4
from experience_replay import ReplayMemory
from agent import Agent, RandomAgent
from Minimax import IA_Minimax

device = "cpu"


class Eval:
    """Evaluation harness pitting a trained DQN agent against Minimax or Random baselines.
    
    Attributes:
        joueur1 (Agent): Evaluated DQN agent set to eval mode.
        joueur2 (IA_Minimax or RandomAgent): Benchmark opponent.
        minimax (int): Depth of Minimax opponent (0 defaults to RandomAgent).
        env (P4Env): Game simulation environment.
        current_player (tuple): Active player and token indicator.
        win_player_1 (int): Total victories recorded for the evaluated agent.
        dir_save (str): Folder destination where winning trajectories are saved.
    """

    def __init__(self, minimax=0, best='Joueur 1'):
        """Initialize evaluation settings and load the target trained model.
        
        Args:
            minimax (int): Minimax search depth (if 0, plays against RandomAgent).
            best (str): Name identifier of the checkpoint to evaluate (e.g. 'Joueur 1').
        """
        self.joueur1 = Agent(name='Joueur 1', hyperparameters_set="puissance4", checkpoint_dir=None)

        self.joueur1.policy_dqn.load_state_dict(state_dict=torch.load(f'checkpoints/puissance4-{best}.pth', map_location=device))
        self.joueur1.policy_dqn.eval()
        if minimax:
            self.joueur2 = IA_Minimax(profondeur_max=minimax)
        else:
            self.joueur2 = RandomAgent(name='Joueur 2')
        
        self.minimax = minimax
        self.env = Puissance4.P4Env()
        self.current_player = (self.joueur1, 1)
        self.win_player_1 = 0
        os.makedirs("evaluations", exist_ok=True)
        self.dir_save = "evaluations/"

    def run(self, episodes=1000):
        """Execute evaluation tournament games and log winning statistics.
        
        Args:
            episodes (int): Number of evaluation games to play (default 1000).
        """
        for _ in range(episodes):
            state = self.env.reset()
            history = [state]
            state = torch.tensor(state, dtype=torch.float, device=device)
            done = False

            while not done:
                if isinstance(self.current_player[0], IA_Minimax):
                    state_np = state.cpu().numpy()
                    action = self.current_player[0].choisir_coup(state_np)
                else:
                    state_perspective = state * self.current_player[1] 
                    action = self.current_player[0].select_action(state=state_perspective, vali_actions=self.env.coups_valides(), is_training=False)
                
                new_state, _, done, info = self.env.step(action, self.current_player[1])
                history.append(new_state)
                
                state = torch.tensor(new_state, dtype=torch.float, device=device)

                if done:
                    break
                
                # Switch player turn
                self.current_player = (self.joueur2, -1) if self.current_player == (self.joueur1, 1) else (self.joueur1, 1)
            
            if self.current_player[0] == self.joueur1 and info['status'][0] == 'Victoire':
                self.win_player_1 += 1
                if self.minimax > 0:
                    torch.save(history, f"{self.dir_save}/history_{self.win_player_1}_D{self.minimax}.csv")

        print(f"The model got {self.win_player_1}/{episodes} wins against {self.joueur2}")


if __name__ == "__main__":
    duel = Eval(minimax=2)
    start_time = time.time()
    duel.run(episodes=1000)
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")