from Minimax import IA_Minimax
from Puissance4 import P4Env
from random import random

class TestMinimax:
    """Interactive CLI testing suite to play Human vs Minimax AI.
    
    Attributes:
        ia (IA_Minimax): Minimax AI agent.
        env (P4Env): Game simulation environment.
        joueur (str): Identifier label for the human participant.
    """

    def __init__(self, profondeur=4):
        """Initialize game environment and Minimax opponent with specified depth.
        
        Args:
            profondeur (int): Search depth limit for Minimax (default 4).
        """
        self.ia = IA_Minimax(profondeur_max=profondeur)
        self.env = P4Env()
        self.joueur = 'humain'

    def test_minimax(self):
        """Launch an interactive terminal game loop pitting a human against Minimax."""
        board = self.env.reset()
        done = False
        current_player = [self.ia, 1] if random() > 0.5 else [self.joueur, 1]
        
        while not done:
            self.env.display()
            
            if current_player[0] == self.joueur:
                while True:
                    try:
                        action = int(input("Choisissez une colonne (0-6) : "))
                        if action in self.env.coups_valides():
                            break
                        print("Colonne invalide, essayez à nouveau.")
                    except ValueError:
                        print("Veuillez entrer un nombre valide.")
            else:
                action = self.ia.choisir_coup(board)
                print(f"L'IA joue la colonne {action}")

            board, _, done, info = self.env.step(action, current_player[1])
            current_player[0] = self.ia if current_player[0] == self.joueur else self.joueur
            current_player[1] *= -1

        self.env.display()
        if info['status'][1] == 1:
            print("Partie terminée, le joueur 1 a gagné !")
        elif info['status'][1] == -1:
            print("Partie terminée, le joueur 2 a gagné !")
        else:
            print("Partie terminée, match nul !")


if __name__ == "__main__":
    test = TestMinimax()
    test.test_minimax()