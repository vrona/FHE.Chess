from piece import *
from square import Square

class Move:

    def __init__(self, source, target):
        
        self.source = source
        self.target = target

    
    def __eq__(self, other): # explicit definition of move equality
        return self.source == other.source and self.target == other.target

    ###### FOR FUTURE REFACTORING ######
    # @staticmethod
    # def move_from_to(source_row, source_col, ok_move_row, ok_move_col):
    #     # micro location
    #     source = Square(source_row, source_col)
    #     target = Square(ok_move_row, ok_move_col)
        
    #     # move at micro
    #     move = Move(source, target)
    #     return move
    #     #piece.add_ok_move(move)