# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:41:05 2024

@author: Richard
"""

import numpy as np

class P4Env:
    """Connect 4 (Puissance 4) game environment following RL conventions.
    
    Attributes:
        lignes (int): Number of rows in the grid (default 6).
        cols (int): Number of columns in the grid (default 7).
        board (np.ndarray): 2D numpy array representing the board state (0: empty, 1: player 1, -1: player 2).
        fini (bool): Flag indicating if the current game has concluded.
        gagnant (int or None): Winning player (1 or -1), or None if game is ongoing/tied.
        incr (int): Move counter since game start.
    """

    def __init__(self):
        """Initialize the game environment dimensions and reset the board state."""
        self.lignes = 6
        self.cols = 7
        self.reset()

    def reset(self):
        """Reset the game board and state variables to start a new match.
        
        Returns:
            np.ndarray: Initial empty board of shape (6, 7).
        """
        self.board = np.zeros((6, 7))
        self.fini = False
        self.gagnant = None
        self.incr = 0
        return self.board
    
    def coups_valides(self):
        """Return a list of playable column indices.
        
        A column is valid if its top row cell is currently unoccupied (0).
        
        Returns:
            list[int]: Column indices (0 to 6) where a token can be legally dropped.
        """
        return [c for c in range(self.cols) if self.board[0, c] == 0]
    
    def step(self, col, joueur):
        """Apply a move for the specified player into the chosen column.
        
        Args:
            col (int): Column index to drop the token into (0-6).
            joueur (int): The current player's token ID (1 or -1).
            
        Returns:
            tuple: (board, reward, done, info)
                board (np.ndarray): The updated game board.
                reward (float): Immediate reward (+10 for victory, 0 for ongoing/tie, -1000 for invalid move).
                done (bool): Whether the game has ended.
                info (dict): Metadata including status message and played row.
        """
        # Validity check
        if col not in self.coups_valides():
            return self.board, -1000, True, {"status": "Invalid Move", 'row': None}
        
        # Drop the token to the lowest unoccupied row
        row_played = -1
        for r in range(self.lignes - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = joueur
                row_played = r
                break
        
        self.incr += 1
        info = {
            "status": "Continue",
            "row": row_played
        }

        # Check win or draw conditions
        if self.verif(joueur):
            self.fini = True
            self.gagnant = joueur
            info['status'] = ('Victoire', joueur)
            return self.board, 10, True, info
        
        if self.incr >= self.lignes * self.cols:
            self.fini = True
            info['status'] = ('Egalité', None)
            return self.board, 0, True, info

        # Game continues
        return self.board, 0, False, info

    def verif(self, joueur):
        """Check if the specified player has aligned 4 tokens.
        
        Evaluates horizontal, vertical, and both diagonal directions.
        
        Args:
            joueur (int): Player ID to check (1 or -1).
            
        Returns:
            bool: True if the player has aligned 4 tokens, False otherwise.
        """
        # Horizontal lines
        for c in range(self.cols - 3):
            for r in range(self.lignes):
                if self.board[r, c] == self.board[r, c + 1] == self.board[r, c + 2] == self.board[r, c + 3] == joueur:
                    return True
            
        # Vertical columns
        for c in range(self.cols):
            for r in range(self.lignes - 3):
                if self.board[r, c] == self.board[r + 1, c] == self.board[r + 2, c] == self.board[r + 3, c] == joueur:
                    return True
        
        # Diagonals (top-left to bottom-right)
        for c in range(self.cols - 3):
            for r in range(self.lignes - 3):
                if self.board[r, c] == self.board[r + 1, c + 1] == self.board[r + 2, c + 2] == self.board[r + 3, c + 3] == joueur:
                    return True

        # Diagonals (bottom-left to top-right)
        for c in range(self.cols - 3):
            for r in range(3, self.lignes):
                if self.board[r, c] == self.board[r - 1, c + 1] == self.board[r - 2, c + 2] == self.board[r - 3, c + 3] == joueur:
                    return True
        
        return False

    def display(self):
        """Print the current board configuration to the standard output."""
        print(self.board)
        print('-' * 20)
