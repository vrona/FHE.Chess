
from base import *
from square import Square

class Board:

    def __init__(self) -> None:
        self.squares = [[0]*8 for col in range(cb_cols)]
        self._create()

    def _create(self):
        # each columns is initialized with 8 zeros related to 8 rows which are actually squares.
        

        for row in range(cb_rows):
            for col in range(cb_cols):
                self.squares[row][col] = Square(row, col)
    
    def _add_pieces(self, color):
        pass