import numpy as np
from gamestate import gamestate


class Engine():
    
    def __init__(self):
        self.game_over = False
        self.heuristic_model = None

    def minimax(self, node: gamestate, depth: int, maximizer: bool):
        
        if depth == 0 or node.is_terminal():
            return node.score(), None #self.heuristic_model
        
        if maximizer:
            value = float('inf')
            moves = node.get_possible_moves()
            
        
        

"""
Analysis chosen move in 1st place

pseudo code

Which input_data?
Alpha Prune
if Type(Child) = Min # minimizer:
    if Uncle(Child) and Weight(Uncle):
        if Weight(Child) <= Weight(Uncle):
            Pruning
            Weight(Dad) = Weight(Child) NNUE Which input_data?
Which output_data?

Beta Prune
if Type(Child) = Max # maximizer:
    if Uncle(Child) and Weight(Uncle):
        if Weight(Child) >= Weight(Uncle):
            Pruning
            Weight(Dad) = Weight(Child) NNUE
Which output_data?

TRI MEMORY PRUNE_MOVE GOOD_MOVE_SEQUENCE PATTERNS
Depth 1: Heuristic sorting
Depth 2: Heuristic sorting
Depth 3: Heuristic sorting

"""

"""
FLOW
encrypted board
Alpha-beta(board):
    
    recursive(simulated.board(move for moves in depth) -> evaluation
    -> best eval, best move

"""

