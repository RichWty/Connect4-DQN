from Minimax import IA_Minimax
from Puissance4 import P4Env
from random import random

class TestMinimax:
    def __init__(self, profondeur = 4):
        self.ia = IA_Minimax(profondeur_max=profondeur)
        self.env = P4Env()
        self.joueur = 'humain'


    def test_minimax(self):
        # Test du fonctionnement de l'IA Minimax
        
        board = self.env.reset()
        done = False
        current_player = [self.ia, 1] if random()>.5 else [self.joueur,1]
        
        while not done:
            self.env.display()
            
            action = -1
            while (current_player[0] == self.joueur) and (action not in self.env.coups_valides()):
                action = int(input("Choisissez une colonne (0-6) : "))
                if action not in self.env.coups_valides():
                    print("Colonne invalide, essayez à nouveau.")
            if current_player[0] == self.ia:
                action = self.ia.choisir_coup(board)
            board,_, done,info = self.env.step(action, current_player[1])
            current_player[0] = self.ia if current_player[0] == self.joueur else self.joueur
            current_player[1]  *= -1

        if info['status'][1] == 1:
            print("Partie terminée, le joueur 1 a gagné !")
        elif info['status'][1] == -1:
            print("Partie terminée, le joueur 2 a gagné !")
        else:
            print("Partie terminée, match nul !")



if __name__ == "__main__":
    test = TestMinimax()
    test.test_minimax()