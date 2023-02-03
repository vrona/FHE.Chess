from square import Square
from piece import *

class Move:

    def __init__(self, initial, destination):
        
        self.initial = initial
        self.destination = destination
    
    def __eq__(self, other): # explicit definition of move equality
        return self.initial == other.initial and self.destination == other.destination

    # @staticmethod
    # def move_from_to(initial_row, initial_col, ok_move_row, ok_move_col):
    #     # micro location
    #     initial = Square(initial_row, initial_col)
    #     destination = Square(ok_move_row, ok_move_col)
        
    #     # move at micro
    #     move = Move(initial, destination)
    #     return move
    #     #piece.add_ok_move(move)