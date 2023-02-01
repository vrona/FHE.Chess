
from base import *
from square import Square
from piece import *
from move import *

class Board:

    def __init__(self):
        # each columns is initialized with 8 zeros related to 8 rows which are actually squares.
        self.squares = [[0]*8 for col in range(cb_cols)]
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')

    
    def compute_move(self, piece, row, col):
        """
        computes possible moves() of specific piece at a given coordinates
        """

        def pawn_moves():
            
            #check initial movement of 2 steps
            steps = 1 if piece.moved else 2

            # vertical movement
            start = row + piece.dir
            end = row + (piece.dir * (1 + steps))
            for possible_move_row in range(start, end, piece.dir):
                if Square.in_board(possible_move_row):
                    if self.squares[possible_move_row][col].empty():

                        # micro location
                        initial = Square(row, col) 
                        destination = Square(possible_move_row, col)
                        
                        # move at micro
                        move = Move(initial, destination)
                        piece.add_ok_move(move)
                    else: break
                else: break

            # attack movement
            possible_move_row = row + piece.dir
            possible_move_col = [col-1, col+1]
            for move_col in possible_move_col:
                if Square.in_board(move_col):
                    if self.squares[possible_move_row][move_col].opponent_presence(piece.color):

                         # micro location
                        initial = Square(row, col) 
                        destination = Square(possible_move_row, move_col)
                        
                        # move at micro
                        move = Move(initial, destination)
                        piece.add_ok_move(move)                                                

            # promotion & en passant

        def kight_moves():
            possible_moves = [
                (row-2, col+1),
                (row-1, col+2),
                (row+1, col+2),
                (row+2, col+1),
                (row+2, col-1),
                (row+1, col-2),
                (row-1, col-2),
                (row-2, col-1),
            ]
            for ok_move in possible_moves:
                ok_move_row, ok_move_col = ok_move # y, x

                # macro location
                if Square.in_board(ok_move_row, ok_move_col):
                    if self.squares[ok_move_row][ok_move_col].empty_occupied(piece.color):
                        
                        # micro location
                        initial = Square(row, col) 
                        destination = Square(ok_move_row, ok_move_col)
                        
                        # move at micro
                        move = Move(initial, destination)
                        piece.add_ok_move(move)


        def straightline_move(): pass

        if isinstance(piece, Pawn): pawn_moves()
        elif isinstance(piece, Knight): kight_moves()
        elif isinstance(piece, Bishop): pass
        elif isinstance(piece, Rook): pass
        elif isinstance(piece, Queen): pass
        elif isinstance(piece, King): pass


    def _create(self):

        for row in range(cb_rows):
            for col in range(cb_cols):
                self.squares[row][col] = Square(row, col)
    
    def _add_pieces(self, color):
        row_pawn, row_other = (6,7) if color == 'white' else (1,0)

        # pawns
        for col in range(cb_cols):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))

        # self.squares[5][0] = Square(5, 0, Pawn(color))
        # self.squares[5][3] = Square(5, 3, Pawn(color))
        # knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))

        # beshops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))

        # rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))
        
        # queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))

        # king
        self.squares[row_other][4] = Square(row_other, 4, King(color))