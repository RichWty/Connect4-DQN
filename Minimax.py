import math
import copy
import numpy as np

class IA_Minimax:
    def __init__(self, profondeur_max = 4):

        self.profondeur_max = profondeur_max
        self.noeuds_explores = 0
        self.transpo = {}

    def minimax(self, board, profondeur, alpha, beta, maximizingPlayer):
        self.noeuds_explores += 1

        boardkey = board.tobytes()

        meilleur_coup_precedent = None

        if boardkey in self.transpo:
            meilleur_coup_precedent = self.transpo[boardkey]['col']
        
        board_miroir = np.fliplr(board) 
        key_miroir = board_miroir.tobytes()

        if key_miroir in self.transpo:
            entry = self.transpo[key_miroir]
            if entry['depth'] >= profondeur:
                if entry['col'] is not None:
                    valeur = entry['value']
                    col_miroir = entry['col']
                    
                    largeur_grille = len(board[0])
                    best_col_reel = (largeur_grille - 1) - col_miroir
                else:
                    valeur = entry['value']
                    best_col_reel = None
                return best_col_reel, valeur
            
        #1 Condition d'arret
        coups_valides = self.coups_valides(board)
        est_fini = self.est_fini(board) 

        if self.coups_gagnant(board, -1):                        # Joueur 2 a gagné
            return (None, 1e10 + profondeur)                    # Valorise le jeu rapide
        if self.coups_gagnant(board, 1):                        # Joueur 1 a gagné
            return (None, -1e10 - profondeur)                   # Valorise le jeu lent
        
        if profondeur == 0 or est_fini:
            if len(coups_valides) == 0:                         # Match nul
                return (None, 0)
            else:
                return (None, self.score_position(board, -1))    # L'heuristique

        best_col = None

        def ordre_de_tri(colonne):
            if colonne == meilleur_coup_precedent:
                return -1000
            return abs(colonne-centre)
         
        # Branche Max
        if maximizingPlayer:
            valeur = -math.inf
            best_col = np.random.choice(coups_valides)

            #Optim : Ordonne les coups pour regarder le centre en premier
            centre = len(board[0]) //2
            coups_valides.sort(key=ordre_de_tri)

            for col in coups_valides:
                b_copy = copy.copy(board)
                self.select_action(b_copy, col, -1)
                nouveau_score = self.minimax(b_copy, profondeur - 1, alpha, beta, False)[1]
                if nouveau_score > valeur:
                    valeur = nouveau_score
                    best_col = col
                alpha = max(alpha, valeur)
                if alpha >= beta:
                    break   # Elagage beta
        
        else:
            # Branche Min
            valeur = math.inf
            best_col = np.random.choice(coups_valides)

            #Optim : Ordonne les coups pour regarder le centre en premier
            centre = len(board[0]) //2
            coups_valides.sort(key=ordre_de_tri)
            for col in coups_valides:
                b_copy = board.copy()
                self.select_action(b_copy, col, 1)
                nouveau_score = self.minimax(b_copy, profondeur - 1, alpha, beta, True)[1]
                if nouveau_score < valeur:
                    valeur = nouveau_score
                    best_col = col
                beta = min(beta, valeur)
                if alpha >= beta:
                    break
        self.transpo[boardkey] = {'depth' : profondeur, 'value' : valeur, 'col': best_col}
        return best_col, valeur


    def coups_valides(self,board):
        # Retourne les coups valides
        return [c for c in range(len(board[0])) if board[0,c] == 0]


    def select_action(self, board, col, player):
        # Place le jeton dans la colonne
        for r in range(len(board)-1, -1, -1):
            if board[r, col] == 0:
                board[r,col] = player
                break
    

    def coups_gagnant(self, board, joueur):
        #Bitboard creation
        bitboard = 0
        for c in range(7):
            for r in range(6):
                if board[r, c] == joueur:
                    bitboard |= (1 << (c*7 + (5-r)))
        
        #verif
        m = bitboard & (bitboard >> 1)
        if m & (m >> 2): return True

        # Décalage horizontal (7 bits)
        m = bitboard & (bitboard >> 7)
        if m & (m >> 14): return True

        # Diagonale 1 (6 bits)
        m = bitboard & (bitboard >> 6)
        if m & (m >> 12): return True

        # Diagonale 2 (8 bits)
        m = bitboard & (bitboard >> 8)
        if m & (m >> 16): return True

        return False

    def est_fini(self, board):
        # Vérifie si la partie est finie
        return self.coups_gagnant(board, 1) or self.coups_gagnant(board, -1) or len(self.coups_valides(board)) == 0

    
    def score_position(self, board, joueur):
        score = 0
        # Heuristique pour connaitre la valeur d'une position
        lignes = len(board)
        cols = len(board[0])
        joueur_adv = 1 if joueur == -1 else -1 # L'adversaire

        # --- 1. Bonus pour la colonne centrale (Stratégie) ---
        # Le centre est crucial au Puissance 4 car il ouvre plus de possibilités
        centre_array = [int(i) for i in list(board[:, cols//2])]
        centre_count = centre_array.count(joueur)
        score += centre_count * 3


        # Lignes
        for c in range(cols - 3):
            for r in range(lignes):
                fenetre = [board[r, c+i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Colonnes
        for c in range(cols):
            for r in range(lignes - 3):
                fenetre = [board[r+i, c] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Diagonales du haut vers le bas
        for c in range(cols - 3):
            for r in range(lignes - 3):
                fenetre = [board[r+i, c+i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Diagonales du bas vers le haut
        for c in range(cols - 3):
            for r in range(3, lignes):
                fenetre = [board[r-i, c+i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        return score
    
    def evaluer_fenetre(self, fenetre, joueur, adversaire): 
        score = 0
        if fenetre.count(joueur) == 4:
            score += 10000
            return score
        elif fenetre.count(joueur) == 3 and fenetre.count(0) == 1:
            score += 10
        elif fenetre.count(joueur) == 2 and fenetre.count(0) == 2:
            score += 2
        
        if fenetre.count(adversaire) == 3 and fenetre.count(0) == 1:
            score -= 40
        
        return score
    
    def choisir_coup(self,board):
        self.noeuds_explores = 0
        meilleur_coup = None

        for prof_actuelle in range(1, self.profondeur_max+1):
            col, score = self.minimax(board, prof_actuelle, -math.inf, math.inf, True)

            meilleur_coup = col

            if score > 9e9 :
                break

        # print(f'Noeuds explorés par Minimax : {self.noeuds_explores}')
        return meilleur_coup
                
        

            
