from base import *
from square import Square
from piece import *
from move import *
import copy

class Board:

    def __init__(self):
        # each columns is initialized with 8 zeros related to 8 rows which are actually squares.
        self.squares = [[0]*8 for col in range(cb_cols)]
        self.last_move = None
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')

    def move(self, piece, move):
        initial = move.initial
        destination = move.destination

        # updating the chessboard
        self.squares[initial.row][initial.col].piece = None
        self.squares[destination.row][destination.col].piece = piece

        # check for pawn promotion
        if isinstance(piece, Pawn):
            self.check_pawn_promotion(piece, destination)

        # check king castling
        if isinstance(piece, King):
            if self.castling(initial, destination):
                diff = destination.col - initial.col
                rook = piece.left_rook if (diff < 0) else piece.right_rook # determine if queen castling or king castling
                self.move(rook, rook.ok_moves[-1])

        # move
        piece.moved = True

        # keep last move
        self.last_move = move

        # clear stock of ok moves
        piece.clear_moves()
    
    def check_pawn_promotion(self, piece, destination):
        if destination.row == 0 or destination.row == 7:
            self.squares[destination.row][destination.col].piece= Queen(piece.color)

    def castling(self, initial, destination):
        return abs(initial.col - destination.col) == 2
        
    def valid_move(self, piece, move):
        return move in piece.ok_moves
    
    def check_simulation(self, piece, move):
        """"for simulation"""
        temppiece = copy.deepcopy(piece)
        tempboard = copy.deepcopy(self)
        tempboard.move(temppiece, move)

        for row in range(cb_rows):
            for col in range(cb_cols):
                if tempboard.squares[row][col].opponent_presence(piece.color):
                    piece = tempboard.squares[row][col].piece
                    tempboard.compute_move(piece, row, col)

                    for mvmt in piece.ok_moves:
                        if isinstance(mvmt.destination.piece, King):
                            return True
        return False

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
                        piece_destination = self.squares[possible_move_row][possible_move_col].piece # get piece at destination aka king check
                        destination = Square(possible_move_row, move_col, piece_destination)
                        
                        # move at micro
                        move = Move(initial, destination)
                        piece.add_ok_move(move)                                                

            # missing en passant

        def kight_moves():
            possible_moves = [
                (row-2, col+1),
                (row-1, col+2),
                (row+1, col+2),
                (row+2, col+1),
                (row+2, col-1),
                (row+1, col-2),
                (row-1, col-2),
                (row-2, col-1)
            ]
            for ok_move in possible_moves:
                ok_move_row, ok_move_col = ok_move # y, x

                # macro location
                if Square.in_board(ok_move_row, ok_move_col):
                    if self.squares[ok_move_row][ok_move_col].empty_occupied(piece.color):
                        
                        piece_destination = self.squares[ok_move_row][ok_move_col].piece # get piece at destination aka king check
                        Move.move_from_to(piece, row, col, ok_move_row, ok_move_col)

        def straightline_move(increments):
            for incr in increments:
                row_incr, col_incr = incr
                possible_move_row = row + row_incr
                possible_move_col = col + col_incr

                while True:
                    if Square.in_board(possible_move_row, possible_move_col):
                        
                        # micro location
                        initial = Square(row, col)
                        piece_destination = self.squares[possible_move_row][possible_move_col].piece # get piece at destination aka king check
                        destination = Square(possible_move_row, possible_move_col, piece_destination)
                            
                        # move at micro
                        move = Move(initial, destination)
                        
                        # empty square
                        if self.squares[possible_move_row][possible_move_col].empty():
                            piece.add_ok_move(move)

                        # opponent presence
                        elif self.squares[possible_move_row][possible_move_col].opponent_presence(piece.color):
                            piece.add_ok_move(move)
                            break
                    
                        # player presence
                        elif self.squares[possible_move_row][possible_move_col].player_presence(piece.color):
                            break

                    else: break 
                    # incrementing_incrs
                    possible_move_row = possible_move_row + row_incr
                    possible_move_col = possible_move_col + col_incr

        def king_moves():
            juxtapose = [
                (row-1, col+0), #N
                (row-1, col+1), #NE
                (row+0, col+1), #E
                (row+1, col+1), #SE
                (row+1, col+0), #S
                (row+1, col-1), #SW
                (row+0, col-1), #W
                (row-1, col-1)] #NW
            
            for ok_move in juxtapose:
                ok_move_row, ok_move_col = ok_move # y, x

                # macro location
                if Square.in_board(ok_move_row, ok_move_col):
                    if self.squares[ok_move_row][ok_move_col].empty_occupied(piece.color):
                        
                        Move.move_from_to(piece, row, col, ok_move_row, ok_move_col)

            # castling
            if not piece.moved:
                
                # queenside
                left_rook = self.squares[row][0].piece

                if isinstance(left_rook, Rook):
                    if not left_rook.moved:
                        for c in range(1, 4):
                            if self.squares[row][c].piece_presence(): # castling abord because of piece presence
                                break

                            if c == 3:
                                piece.left_rook = left_rook # adds left rook to queen
                                
                                # rook move to king
                                Move.move_from_to(left_rook, row, 0, row, 3)

                                # king move to rook
                                Move.move_from_to(piece, row, col, row, 2)

                # kingside
                right_rook = self.squares[row][7].piece

                if isinstance(right_rook, Rook):
                    if not right_rook.moved:
                        for c in range(5, 7):
                            if self.squares[row][c].piece_presence(): # castling abord because of piece presence
                                break

                            if c == 6:
                                piece.right_rook = right_rook # adds right rook to king
                                
                                # rook move to king
                                Move.move_from_to(right_rook, row, 7, row, 5)

                                # king move to rook
                                Move.move_from_to(piece, row, col, row, 6)


        if isinstance(piece, Pawn): pawn_moves()

        elif isinstance(piece, Knight): kight_moves()

        elif isinstance(piece, Bishop):
            straightline_move([
                (-1,1), #to NE
                (-1,-1),#to NW
                (1,-1), #to SW
                (1,1)   #to SE
            ])

        elif isinstance(piece, Rook):
            straightline_move([
                (-1,0),#N
                (0,1), #E
                (0,-1),#W
                (1,0)  #S
            ])

        elif isinstance(piece, Queen):
            straightline_move([
                (-1,0),#N
                (0,1), #E
                (0,-1),#W
                (1,0), #S
                (-1,1), #to NE
                (-1,-1),#to NW
                (1,-1), #to SW
                (1,1)   #to SE
            ])
        
        elif isinstance(piece, King): king_moves()

    def _create(self):

        for row in range(cb_rows):
            for col in range(cb_cols):
                self.squares[row][col] = Square(row, col)
    
    def _add_pieces(self, color):
        row_pawn, row_other = (6,7) if color == 'white' else (1,0)

        # pawns
        for col in range(cb_cols):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))
        # test self.squares[5][0] = Square(5, 0, Pawn(color))
        # test self.squares[5][3] = Square(5, 3, Pawn(color))
        
        # knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))

        # bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))
        # test self.squares[4][3] = Square(4, 3, Bishop(color))
        
        # rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))
        # test self.squares[row_other][7] = Square(row_other, 7, Rook(color))
        
        # queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))
        # test self.squares[3][3] = Square(3, 3, Queen(color))

        # king
        self.squares[row_other][4] = Square(row_other, 4, King(color))
        # testself.squares[2][4] = Square(2, 4, King(color))