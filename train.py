import os
import sys
import yaml
import torch
import time
from random import choice

import Puissance4
from experience_replay import ReplayMemory
from agent import Agent
from eval import Eval

device = "cuda" if torch.cuda.is_available() else "cpu"


class Duel:

    def __init__(self, load = True):
        self.joueur1 = Agent(name='Joueur 1', hyperparameters_set="puissance4", checkpoint_dir=None)
        self.joueur2 = Agent(name='Joueur 2', hyperparameters_set="puissance4", checkpoint_dir=None)

        if load :
            self.joueur1.policy_dqn.load_state_dict(state_dict=torch.load(f'checkpoints/puissance4-Joueur 1.pth', map_location=device))
            self.joueur2.policy_dqn.load_state_dict(state_dict=torch.load(f'checkpoints/puissance4-Joueur 2.pth', map_location=device))

        self.env = Puissance4.P4Env()

        self.replay_memory_1 = ReplayMemory(capacity=100000)
        self.replay_memory_2 = ReplayMemory(capacity=100000)

        self.current_player = (self.joueur1, 1)

        self.minimax_version = 'won_against_minimax'
        os.makedirs(self.minimax_version, exist_ok=True)

    def run(self, episodes):

        win_player_1 = 0
        list_win = [0]
        list_update = [0]
        win_player_2 = 0
        best_eval_score = -1
        minimax = 1

        step_count = 0
        for episode in range(1,episodes+1):
            self.env.reset()
            done = False
            play = 0

            while play<3 and minimax == 1:
                self.env.step(col = choice(range(self.env.cols)), joueur = self.current_player[1])
                self.current_player = (self.joueur2, -1) if self.current_player == (self.joueur1, 1) else (self.joueur1, 1)
                play+=1
            
            # Creation du tensor de l'état après les coups aléatoires
            state = torch.tensor(self.env.board, dtype=torch.float, device=device)

            memoire_temporaire1 = []
            memoire_temporaire2 = []

            while not done:
                state_perspective = state * self.current_player[1]  # Invert state for perspective
                action = self.current_player[0].select_action(state = state_perspective, vali_actions=self.env.coups_valides(), is_training=True)
                new_state, reward, done, info = self.env.step(action, self.current_player[1])

                new_state_tensor = torch.tensor(new_state, dtype=torch.float, device=device)
                
                # L'état suivant est évalué du point de vue de l'adversaire
                new_state_opp = new_state_tensor * (-self.current_player[1])

                reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)
                action_tensor = torch.tensor(action, dtype=torch.long, device=device)
                done_tensor = torch.tensor(done, dtype=torch.float, device=device)
                
                next_valid_actions = self.env.coups_valides()
                next_valid_mask = torch.zeros(7, dtype=torch.bool, device=device)
                if not done:
                    next_valid_mask[next_valid_actions] = True

                if self.current_player == (self.joueur1, 1):
                    memoire_temporaire1.append((state_perspective, action_tensor, reward_tensor, new_state_opp, next_valid_mask, done_tensor))
                else:
                    memoire_temporaire2.append((state_perspective, action_tensor, reward_tensor, new_state_opp, next_valid_mask, done_tensor))

                step_count+=1
                state = new_state_tensor # Update state to the new true state

                if done:
                    break
                
                # Switch player
                self.current_player = (self.joueur2, -1) if self.current_player == (self.joueur1, 1) else (self.joueur1, 1)

            if episode % 100 == 0:
                print(f"Episode {episode}: Player  wins: {list_win[-1]}")

            if self.env.gagnant == 1:
                win_player_1 += 1
            elif self.env.gagnant == -1:
                win_player_2 += 1

            list_win.append(win_player_1-win_player_2)

            self.replay_memory_1.extend(memoire_temporaire1)
            self.replay_memory_2.extend(memoire_temporaire2)

            #plot_wins(list_win)

            self.joueur1.update_epsilon()
            self.joueur2.update_epsilon()

            if abs(list_win[-1]) > 50:
                current_player = self.joueur1 if list_win[-1] > 0 else self.joueur2
                log_message = f"Player {'1' if list_win[-1] > 0 else '2'} is dominating with a score of {list_win[-1]} at episode {episode+1}"
                print(log_message)
                list_update.append((abs(list_win[-1]))) # pyright: ignore[reportArgumentType]
                with open(current_player.LOG_FILE, 'a') as log_file:
                    log_file.write(log_message + '\n')

                torch.save(current_player.policy_dqn.state_dict(), current_player.MODEL_FILE)
                if current_player == self.joueur1:
                    self.joueur2.policy_dqn.load_state_dict(self.joueur1.policy_dqn.state_dict())
                else:
                    self.joueur1.policy_dqn.load_state_dict(self.joueur2.policy_dqn.state_dict())

                self.joueur1.update_target_network()
                self.joueur2.update_target_network()

                win_player_1 = win_player_2 = 0
                list_win = [0]

            if len(self.replay_memory_1) > self.current_player[0].batch_size:
                mini_batch_1 = self.replay_memory_1.sample(self.current_player[0].batch_size)
                mini_batch_2 = self.replay_memory_2.sample(self.current_player[0].batch_size)
                self.joueur1.optimize(mini_batch_1, self.joueur1.policy_dqn, self.joueur1.target_dqn)
                self.joueur2.optimize(mini_batch_2, self.joueur2.policy_dqn, self.joueur2.target_dqn)

                if step_count > self.joueur1.network_update_freq:
                    self.joueur1.update_target_network()
                    self.joueur2.update_target_network()

            if episode%10000 == 0:
                print(f"Épisode {episode}: {list_win[-1]}")
                score_total = 0
                best = 'Joueur 1' if list_win[-1] > 0 else 'Joueur 2'
                Best = self.joueur1 if best == 'Joueur 1' else self.joueur2
                if minimax==1:
                    eval_random = Eval(minimax=0, best=best)
                    eval_random.run(episodes=100)

                if eval_random.win_player_1 / 100 > 0.85 or minimax > 1:
                    print(f"Playing against minimax D{minimax}...")
                    eval_minimax = Eval(minimax=minimax, best=best)
                    eval_minimax.run(episodes=100)
                    score_total = eval_minimax.win_player_1
                    if eval_minimax.win_player_1 / 100 > 0.8:
                        print(f"🎉 L'agent a atteint un taux de victoire de {eval_minimax.win_player_1 / 100:.2%} contre Minimax D{minimax} !")
                        print("On augmente le niveau de minimax")
                        
                        torch.save(Best.policy_dqn.state_dict(), self.minimax_version + f"/best_against_minimax_D{minimax}.pth")

                        minimax += 1

                score_total += eval_random.win_player_1
                    
                if score_total > best_eval_score:
                    best_eval_score = score_total
                    print(f"🏆 Nouveau record en évaluation ! Score : {score_total}/200. Sauvegarde du modèle...")
                    
                    # On récupère l'agent qui vient d'être évalué
                    agent_evalue = self.joueur1 if best == 'Joueur 1' else self.joueur2
                    
                    # On le sauvegarde sous un nom spécifique
                    torch.save(agent_evalue.policy_dqn.state_dict(), "best_eval_model.pth")
                    
                win_player_1 = win_player_2 = 0


if __name__ == "__main__":
    duel = Duel(load=False)
    duel.run(episodes=100000000)