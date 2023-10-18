from base import *
from piece import *
from move import *
from square import Square
from clone_chess import Clone_Chess

class Board:

    def __init__(self):
        # each columns is initialized with 8 zeros related to 8 rows which are actually squares.
        self.squares = [[0]*8 for col in range(cb_cols)]
        self.last_move = None
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')
        self.clone_chess = Clone_Chess()

    def move(self, piece, move, simulation = False):
        source = move.source
        target = move.target

        # setting en_passant necessity
        en_passant_empty = self.squares[target.row][target.col].empty()

        # updating the chessboard
        self.squares[source.row][source.col].piece = None
        self.squares[target.row][target.col].piece = piece

        # check for pawn promotion
        if isinstance(piece, Pawn):
            # en_passant capture
            diff = target.col - source.col

            if diff != 0 and en_passant_empty:
                 # updating the chessboard
                self.squares[source.row][source.col + diff].piece = None
                self.squares[target.row][target.col].piece = piece
            
            else:
                # promotion check
                self.check_pawn_promotion(piece, target)
                #self.squares[source.row][source.col].piece.promoted = True

        # check king castling
        if isinstance(piece, King):
            if self.castling(source, target) and not simulation:
                diff = target.col - source.col

                if (diff < 0):
                    rook = piece.left_rook # castling queenside
                    self.move(rook, rook.legal_move[0])
                else :
                    rook = piece.right_rook # castling kingside
                    self.move(rook, rook.legal_move[-1])

        
        # move
        piece.moved = True

        # keep last move
        self.last_move = move

        # clear stock of ok moves
        #piece.clear_moves()
        
        ## clearing all the pieces legal_move
        for row in range(cb_rows):
            for col in range(cb_cols):
                if self.squares[row][col].piece_presence():
                    p = self.squares[row][col].piece
                    p.clear_legal_move()

    
    def check_pawn_promotion(self, piece, target):
        if target.row == 0 or target.row == 7:
            self.squares[target.row][target.col].piece= Queen(piece.color)
            self.squares[target.row][target.col].piece.is_promotion = True

    def castling(self, source, target):
        return abs(source.col - target.col) == 2

    def set_true_en_passant(self, piece):
        
        if not isinstance(piece, Pawn):
            return

        for row in range(cb_rows):
            for col in range(cb_cols):
                if isinstance(self.squares[row][col].piece, Pawn):
                    self.squares[row][col].piece.en_passant = False
        
        piece.en_passant = True 

    # check if move is valid (not based on chess lib)
    def new_valid_move(self, piece, move):
        return move in piece.legal_move  


    def piece_legal(self, current_board, piece):
        """
        legal: several proposal > for i in proposal, get source + target then push move(proposal) add_ok_moves
        """
        # make a list of legal moves from python-chess
        list_legal = list(current_board.legal_moves)

        # make a dict of legal moves where keys is a tuple of splitted sprint from keys
        coordinate_legal = {tuple(alphasq):[] for alphasq in alphanum_square.keys()}
        
        for i, lv in enumerate(list_legal):
            coordinate_legal[tuple(str(list_legal[i])[:2])].append(tuple(str(lv)[2:]))

        for source_alpha, target_list_tuple in coordinate_legal.items():

            source = Square(8 -int(source_alpha[1]), Square.convert_algeb_not(source_alpha[0]))

            for target_alpha in target_list_tuple: #('g', '1', 'f', '3')
                target = Square(8 -int(target_alpha[1]), Square.convert_algeb_not(target_alpha[0]))
    
                move = Move(source, target)
        
                if self.squares[8 -int(source_alpha[1])][Square.convert_algeb_not(source_alpha[0])].piece_type() is not None: #and current_board.is_check() != True

                    self.squares[8 -int(source_alpha[1])][Square.convert_algeb_not(source_alpha[0])].piece.add_legalmove(move)

    def compute_move(self, piece, row, col, bool=True):
        """
        computes possible moves() of specific piece at a given coordinates
        """
    
        def king_moves():


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
                                # micro location
                                source = Square(row, 0)
                                target = Square(row, 3)

                                # move at micro
                                rook_move = Move(source, target)
                                
                                # king move to rook
                                # micro location
                                source = Square(row, col)
                                target = Square(row, 2)

                                # move at micro                                
                                left_rook.add_legalmove(rook_move)


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
                                    # micro location
                                source = Square(row, 7)
                                target = Square(row, 5)

                                    # move at micro
                                rook_move = Move(source, target)
                                
                                # king move to rook
                                    # micro location
                                source = Square(row, col)
                                target = Square(row, 6)
                                
                                    # move at micro
                                right_rook.add_legalmove(rook_move)

        
        if isinstance(piece, King): king_moves()


    def _create(self):
        """At step 0 (beginning or reset), board is created"""
        for row in range(cb_rows):
            for col in range(cb_cols):
                self.squares[row][col] = Square(row, col)
    

    def _add_pieces(self, color):
        """At step 0 (beginning or reset), pieces are created"""

        row_pawn, row_other = (6,7) if color == 'white' else (1,0)

        # pawns
        for col in range(cb_cols):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))

        # knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))

        # bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))
        
        # rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))
        
        # queen
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))

        # king
        self.squares[row_other][4] = Square(row_other, 4, King(color))