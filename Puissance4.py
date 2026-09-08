# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:41:05 2024

@author: Richard
"""

import numpy as np

class P4Env:
    #Le jeu de puissance 4
    def __init__(self):
        #Initialisation du jeu
        self.lignes = 6
        self.cols = 7
        self.reset()

    def reset(self):
        # Reinitialise le jeu
        self.board = np.zeros((6, 7))
        self.fini = False
        self.gagnant = None
        self.incr = 0
        return self.board
    
    def coups_valides(self):
        # Retourne les coups valides
        return [c for c in range(self.cols) if self.board[0,c] == 0]
    
    def step(self,col,joueur):
        """
        Arguments :
            col     :   colonne à jouer
            joueur  :   1 ou -1 
        Retourne :
            board, reward, done
        """

        # Valididté
        if col not in self.coups_valides():
            return self.board, -1000, True, {"status": "Invalid Move", 'row': None}
        
        # Place le jeton
        row_played = -1
        for r in range(self.lignes-1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r,col] = joueur
                row_played = r
                break
        
        self.incr += 1
        info = {
            "status"   :    "Continue",
            "row"      :    row_played
        }

        # Vérifie victoire ou match nul
        if self.verif(joueur):
            self.fini = True
            self.gagnant = joueur
            info['status'] = ('Victoire', joueur)
            return self.board, 10, True, info
        
        if self.incr >= self.lignes * self.cols:
            self.fini = True
            info['status'] = ('Egalité', None)
            return self.board, 0, True, info
        # Partie continue
        return self.board, 0, False, info

    def verif(self, joueur):
        #Fonction pour vérifier si un joueur a gagné
        # lignes
        for c in range(self.cols - 3):
            for r in range(self.lignes):
                if self.board[r, c] == self.board[r, c + 1] == self.board[r, c + 2] == self.board[r, c + 3] == joueur:
                    return True
            
        # colonnes
        for c in range(self.cols):
            for r in range(self.lignes - 3):
                if self.board[r, c] == self.board[r + 1, c] == self.board[r + 2, c] == self.board[r + 3, c] == joueur:
                    return True
        
        # les diagonales sont vérifiées de la gauche vers la droite
        # diagonales du haut vers le bas
        for c in range(self.cols - 3):
            for r in range(self.lignes - 3):
                if self.board[r,c] == self.board[r+1,c+1] == self.board[r+2,c+2] == self.board[r+3,c+3] == joueur:
                    return True

        # les diagonales sont vérifiées de la droite vers la gauche
        # diagonales du haut vers le bas
        for c in range(self.cols - 3):
            for r in range(3, self.lignes):
                if self.board[r,c] == self.board[r-1,c+1] == self.board[r-2,c+2] == self.board[r-3,c+3] == joueur:
                    return True
        
        return False

    def display(self):
        # Fonction optionnelle pour afficher le plateau
        print(self.board)
        print('-'*20)
