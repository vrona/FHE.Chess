import numpy as np

"""from piece class variable legal moves are in self.ok_moves = []"""

def directmove_checkmate(board):
    board = board.copy() # see also simulation in board.py
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        board.push_uci(str(move))

        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()

def distribution_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    # normalization
    probs = probs / probs.sum()
    # increase the gap within distribution with power of 3
    probs = probs ** 3
    # normalization
    probs = probs / probs.sum()
    return probs