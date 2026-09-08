import torch

from Puissance4 import P4Env
from agent import Agent
from Minimax import IA_Minimax

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Test:
    def __init__(self, game = 1, load = None, minimax = 0):

        self.joueur1 = Agent(name='Joueur 1', hyperparameters_set="puissance4", checkpoint_dir=None)
        self.joueur2 = Agent(name='Joueur 2', hyperparameters_set="puissance4", checkpoint_dir=None)
        if load:
            self.joueur1.policy_dqn.load_state_dict(torch.load('checkpoints/puissance4-Joueur 1.pth'))
            self.joueur2.policy_dqn.load_state_dict(torch.load('checkpoints/puissance4-Joueur 2.pth'))
        
        if minimax:
            self.joueur2 = IA_Minimax(profondeur_max=minimax)

        self.game = game

    def run_test(self):
        Env = P4Env()

        for _ in range(self.game):
            state = Env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            current_player = (self.joueur1, 1)
            done = False
            while not done:
                if isinstance(current_player[0], IA_Minimax):
                    state = state.numpy()
                    action = current_player[0].choisir_coup(state)
                    state = torch.tensor(state, dtype=torch.float, device=device)
                else:
                    state *= current_player[1]
                    action = current_player[0].select_action(state=state, vali_actions=range(Env.cols), is_training=False)
                    if action not in Env.coups_valides():
                        print(f"Colonne {action} invalide. L'agent est complettement con.")
                        break
                new_state, _, done, _ = Env.step(action, current_player[1])

                state = torch.tensor(data=new_state, dtype=torch.float, device=device)
                print(f"Player {current_player} played column {action}.")
                Env.display()

                if done:
                    print(f"Game finished ! Winner is Player {current_player[0]}")
                    break

                current_player = (self.joueur2, -1) if current_player == (self.joueur1, 1) else (self.joueur1, 1)


if __name__ == "__main__":
    test = Test(game=1, load=False, minimax=4)
    test.run_test()
