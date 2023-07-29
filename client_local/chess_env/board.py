from base import *
from piece import *
from move import *
import copy
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
                rook = piece.left_rook if (diff < 0) else piece.right_rook # determine if castling queenside or kingside

                self.move(rook, rook.ok_moves[-1])

        # move
        piece.moved = True

        # keep last move
        self.last_move = move

        # clear stock of ok moves
        piece.clear_moves()

    
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
    def valid_move(self, piece, move):
        return move in piece.ok_moves  

 
    # here it simulates if King is check, thus it blocks any movement that lead king to be check.
    # improvements needed cause some deadends.
    def king_check_sim(self, piece, move):

        """"for simulation"""
        temppiece = copy.deepcopy(piece)
        tempboard = copy.deepcopy(self)
        tempboard.move(temppiece, move, simulation=True) # move virtually one piece

        for row in range(cb_rows):
            for col in range(cb_cols):
                """check for all opponent if their potential ok_moves arrive in the team's Kings' square"""
                if tempboard.squares[row][col].opponent_presence(piece.color):
                    p = tempboard.squares[row][col].piece
                    
                    tempboard.compute_move(p, row, col, bool=False)

                    for mvmt in p.ok_moves:
                        if isinstance(mvmt.target.piece, King):
                            return True
        return False


    def sim_kingcheck_okmoves(self, piece, move, bool):
        """adds move into ok_move list if my King is not in check"""
        if bool:
            if not self.king_check_sim(piece, move): # if not in check go ahead
                piece.add_ok_move(move)

        else:
            piece.add_ok_move(move) # if not in check go ahead


    def move_kingchecksim(self, source, target, piece, bool):
        """integrates move into sim_kingcheck_okmoves()"""
        # move at micro
        move = Move(source, target)
        self.sim_kingcheck_okmoves(piece, move, bool)
        

    def compute_move(self, piece, row, col, bool=True):
        """
        computes possible moves() of specific piece at a given coordinates
        """

        def pawn_moves():
            
            #check source movement of 2 steps
            steps = 1 if piece.moved else 2

            # vertical movement
            start = row + piece.dir
            end = row + (piece.dir * (1 + steps))
            for possible_move_row in range(start, end, piece.dir):
                if Square.in_board(possible_move_row):
                    if self.squares[possible_move_row][col].empty():

                        # micro location
                        source = Square(row, col) 
                        target = Square(possible_move_row, col)
                        
                        self.move_kingchecksim(source, target, piece, bool)
                            
                    else: break # move done
                else: break # outside chessboard

            # attack movement
            possible_move_row = row + piece.dir
            possible_move_col = [col-1, col+1]
            for move_col in possible_move_col:
                if Square.in_board(move_col):
                    if self.squares[possible_move_row][move_col].opponent_presence(piece.color):

                        # micro location
                        source = Square(row, col)
                        piece_target = self.squares[possible_move_row][move_col].piece # get piece at target aka king check
                        target = Square(possible_move_row, move_col, piece_target)
                        
                        self.move_kingchecksim(source, target, piece, bool)

            # en_passant
            attacker_ini_row = 3 if piece.color == 'white' else 4
            attacker_desti_row = 2 if piece.color == 'white' else 5

            # left juxtapose square
            if Square.in_board(col-1) and row == attacker_ini_row:
                if self.squares[row][col-1].opponent_presence(piece.color):
                    p = self.squares[row][col-1].piece
                    
                    if isinstance(p, Pawn):

                        if p.en_passant:

                            source = Square(row, col)
                            target = Square(attacker_desti_row, col-1, p)

                            self.move_kingchecksim(source, target, piece, bool)

            # right juxtapose square
            if Square.in_board(col+1) and row == attacker_ini_row:
                if self.squares[row][col+1].opponent_presence(piece.color):
                    p = self.squares[row][col+1].piece
                    
                    if isinstance(p, Pawn):

                        if p.en_passant:

                            source = Square(row, col)
                            target = Square(attacker_desti_row, col+1, p)

                            self.move_kingchecksim(source, target, piece, bool)


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
                        
                        source = Square(row, col)
                        piece_target = self.squares[ok_move_row][ok_move_col].piece # get piece at target aka king check
                        target = Square(ok_move_row, ok_move_col, piece_target)

                        self.move_kingchecksim(source, target, piece, bool)



        def straightline_move(increments):
            for incr in increments:
                row_incr, col_incr = incr
                possible_move_row = row + row_incr
                possible_move_col = col + col_incr

                while True:
                    if Square.in_board(possible_move_row, possible_move_col):
                        
                        # micro location
                        source = Square(row, col)
                        piece_target = self.squares[possible_move_row][possible_move_col].piece # get piece at target aka king check
                        target = Square(possible_move_row, possible_move_col, piece_target)
                            
                        # move at micro
                        move = Move(source, target)
                        
                        # empty square
                        if self.squares[possible_move_row][possible_move_col].empty():
                            
                            self.sim_kingcheck_okmoves(piece,move,bool)

                        # opponent presence
                        elif self.squares[possible_move_row][possible_move_col].opponent_presence(piece.color):
                            
                            self.sim_kingcheck_okmoves(piece,move,bool)

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
                    
                        
                    # micro location
                    source = Square(row, col)
                    target = Square(ok_move_row, ok_move_col)
                    
                    # move at micro
                    move = Move(source, target)
                
                    if self.squares[ok_move_row][ok_move_col].empty_occupied(piece.color):
                        self.sim_kingcheck_okmoves(piece,move,bool)

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
                                king_move = Move(source, target)
                                
                                if bool:
                                    if not self.king_check_sim(left_rook, rook_move) and not self.king_check_sim(piece, king_move): # if not in check go ahead
                                        left_rook.add_ok_move(rook_move)
                                        piece.add_ok_move(king_move)
                                else:
                                        left_rook.add_ok_move(rook_move)
                                        piece.add_ok_move(king_move) # if not in check go ahead


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
                                king_move = Move(source, target)
                                
                                if bool:
                                    if not self.king_check_sim(right_rook, rook_move) and not self.king_check_sim(piece, king_move): # if not in check go ahead
                                        right_rook.add_ok_move(rook_move)
                                        piece.add_ok_move(king_move)
                                else:
                                        right_rook.add_ok_move(rook_move)
                                        piece.add_ok_move(king_move) # if not in check go ahead


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