import math
import copy
import numpy as np

class IA_Minimax:
    """Minimax AI agent with Alpha-Beta pruning, transposition table, and bitboard evaluations.
    
    Attributes:
        profondeur_max (int): Maximum depth limit for the iterative deepening minimax search.
        noeuds_explores (int): Counter tracking nodes visited during search.
        transpo (dict): Transposition table mapping board hash keys to precomputed search results.
    """

    def __init__(self, profondeur_max=4):
        """Initialize the Minimax agent.
        
        Args:
            profondeur_max (int): Max search depth limit (default 4).
        """
        self.profondeur_max = profondeur_max
        self.noeuds_explores = 0
        self.transpo = {}

    def minimax(self, board, profondeur, alpha, beta, maximizingPlayer):
        """Recursive alpha-beta minimax search with transposition tables and horizontal symmetry lookup.
        
        Args:
            board (np.ndarray): Current 6x7 board state.
            profondeur (int): Remaining search depth.
            alpha (float): Lower bound for alpha-beta pruning.
            beta (float): Upper bound for alpha-beta pruning.
            maximizingPlayer (bool): True if maximizing node (player -1), False if minimizing (player 1).
            
        Returns:
            tuple[int or None, float]: (best_column, evaluated_score).
        """
        self.noeuds_explores += 1

        boardkey = board.tobytes()
        meilleur_coup_precedent = None

        if boardkey in self.transpo:
            meilleur_coup_precedent = self.transpo[boardkey]['col']
        
        # Horizontal reflection check for symmetrical evaluation reuse
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
            
        # Stopping conditions
        coups_valides = self.coups_valides(board)
        est_fini = self.est_fini(board) 

        if self.coups_gagnant(board, -1):                        # Player 2 / AI won
            return (None, 1e10 + profondeur)                    # Reward fast wins
        if self.coups_gagnant(board, 1):                        # Player 1 won
            return (None, -1e10 - profondeur)                   # Penalize fast losses
        
        if profondeur == 0 or est_fini:
            if len(coups_valides) == 0:                         # Tie game
                return (None, 0)
            else:
                return (None, self.score_position(board, -1))   # Heuristic evaluation

        best_col = None

        def ordre_de_tri(colonne):
            """Move-ordering heuristic prioritizing previously best moves and central columns."""
            if colonne == meilleur_coup_precedent:
                return -1000
            return abs(colonne - centre)
         
        centre = len(board[0]) // 2

        # Maximizing branch (Player -1)
        if maximizingPlayer:
            valeur = -math.inf
            best_col = np.random.choice(coups_valides)

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
                    break   # Beta cutoff
        
        else:
            # Minimizing branch (Player 1)
            valeur = math.inf
            best_col = np.random.choice(coups_valides)

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
                    break   # Alpha cutoff

        self.transpo[boardkey] = {'depth': profondeur, 'value': valeur, 'col': best_col}
        return best_col, valeur

    def coups_valides(self, board):
        """Return indices of columns that have at least one empty cell.
        
        Args:
            board (np.ndarray): Current board state.
            
        Returns:
            list[int]: Playable column indices.
        """
        return [c for c in range(len(board[0])) if board[0, c] == 0]

    def select_action(self, board, col, player):
        """Drop a piece into the specified column in place.
        
        Args:
            board (np.ndarray): Board array to update.
            col (int): Target column index.
            player (int): Token ID (1 or -1).
        """
        for r in range(len(board) - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = player
                break

    def coups_gagnant(self, board, joueur):
        """Fast win check using bitboard bitwise shift operations.
        
        Args:
            board (np.ndarray): Current board state.
            joueur (int): Player ID to check (1 or -1).
            
        Returns:
            bool: True if player has aligned 4 tokens, False otherwise.
        """
        bitboard = 0
        for c in range(7):
            for r in range(6):
                if board[r, c] == joueur:
                    bitboard |= (1 << (c * 7 + (5 - r)))
        
        # Vertical alignment (1 bit shift)
        m = bitboard & (bitboard >> 1)
        if m & (m >> 2):
            return True

        # Horizontal alignment (7 bits shift)
        m = bitboard & (bitboard >> 7)
        if m & (m >> 14):
            return True

        # Diagonal 1 (\ direction: 6 bits shift)
        m = bitboard & (bitboard >> 6)
        if m & (m >> 12):
            return True

        # Diagonal 2 (/ direction: 8 bits shift)
        m = bitboard & (bitboard >> 8)
        if m & (m >> 16):
            return True

        return False

    def est_fini(self, board):
        """Check if the game has ended by victory or board saturation.
        
        Args:
            board (np.ndarray): Current board state.
            
        Returns:
            bool: True if terminal, False otherwise.
        """
        return self.coups_gagnant(board, 1) or self.coups_gagnant(board, -1) or len(self.coups_valides(board)) == 0
    
    def score_position(self, board, joueur):
        """Calculate heuristic evaluation score for a board position.
        
        Args:
            board (np.ndarray): Board array to evaluate.
            joueur (int): Target player to evaluate for.
            
        Returns:
            int: Positional heuristic score.
        """
        score = 0
        lignes = len(board)
        cols = len(board[0])
        joueur_adv = 1 if joueur == -1 else -1

        # Central column control bonus
        centre_array = [int(i) for i in list(board[:, cols // 2])]
        centre_count = centre_array.count(joueur)
        score += centre_count * 3

        # Horizontal window evaluations
        for c in range(cols - 3):
            for r in range(lignes):
                fenetre = [board[r, c + i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Vertical window evaluations
        for c in range(cols):
            for r in range(lignes - 3):
                fenetre = [board[r + i, c] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Diagonal (down-right) evaluations
        for c in range(cols - 3):
            for r in range(lignes - 3):
                fenetre = [board[r + i, c + i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        # Diagonal (up-right) evaluations
        for c in range(cols - 3):
            for r in range(3, lignes):
                fenetre = [board[r - i, c + i] for i in range(4)]
                score += self.evaluer_fenetre(fenetre, joueur, joueur_adv)
        
        return score
    
    def evaluer_fenetre(self, fenetre, joueur, adversaire): 
        """Score a 4-cell sliding window based on player and opponent tokens.
        
        Args:
            fenetre (list[int]): 4 board values in a line.
            joueur (int): Target player ID.
            adversaire (int): Opponent player ID.
            
        Returns:
            int: Window score contribution.
        """
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
    
    def choisir_coup(self, board):
        """Execute iterative deepening minimax search to select the best action.
        
        Args:
            board (np.ndarray): Current game board.
            
        Returns:
            int: Best column chosen by the Minimax algorithm.
        """
        self.noeuds_explores = 0
        meilleur_coup = None

        for prof_actuelle in range(1, self.profondeur_max + 1):
            col, score = self.minimax(board, prof_actuelle, -math.inf, math.inf, True)
            meilleur_coup = col

            if score > 9e9:  # Winning move found, stop early
                break

        return meilleur_coup
