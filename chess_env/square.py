
class Square:

    algebraic_notation_cols = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}
    digit_notation_cols = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7}

    def __init__(self, row, col, piece=None):
        self.row = row
        self.col = col
        self.piece = piece
        self.algebraic_cols = self.algebraic_notation_cols[col]

    def __eq__(self, other): # explicit definition of square equality
        return self.row == other.row and self.col == other.col
    
    def piece_presence(self):
        return self.piece != None

    def empty(self):
        return not self.piece_presence()

    def player_presence(self, color):
        return self.piece_presence() and self.piece.color == color

    def opponent_presence(self, color):
        return self.piece_presence() and self.piece.color != color

    def empty_occupied(self, color):
        return self.empty() or self.opponent_presence(color)

    @staticmethod
    def in_board(*args):
        for arg in args:
            if arg < 0 or arg > 7:
                return False
        
        return True

    @staticmethod
    def get_algeb_not(col):
        algebraic_notation_cols = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}
        return algebraic_notation_cols[col]
    
    @staticmethod
    def convert_algeb_not(col):
        digit_notation_cols = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7}
        return digit_notation_cols[col]