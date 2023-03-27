import chess
from square import Square
from base import *
#-----python-chess-----#

class Clone_Chess:
    
    def __init__(self):
        self.board = chess.Board()

   #### ACTIONS ####
    
    # make a move from source to target square
    def move_to_board(self, source,target):
        uci_move = "".join((source,target))
        self.board.push_san(uci_move)
        print(self.board)

    def undo_move(self):
            self.board.pop()

    # clearing the board
    def clear_board(self):
        self.board.clear_board()


    #### GETS ####
    # get a snapshot of board
    def get_board(self):
        return self.board
    
    # TO DO get the legal moves for a given position
    def legal_moves(self):
        return self.board.legal_moves
    
    # get the number of legal moves
    def legal_moves_count(self):
        return self.board.legal_moves.count()

    # get game outcome
    def outcome(self):
        return self.board.outcome()
    
    # def get_alpha_location(self, source, target):
    #     for row in range(cb_rows):
    #         for col in range(cb_cols):
    #             print(Square.get_algeb_not(col), row)

    #### CHECKINGS ####

    # check if move within legal moves
    def check_legal_move(self, move):

        self.convert_move_2_string(move)
        
        # for row in range(cb_rows):
        #     for col in range(cb_cols):
        #     source = Square(row, col) 
        #     target = Square(possible_move_row, col)
        #print(source.row, source.col)
        # uci_format = "".join((source,target))
        
        # print(self.legal_moves(), chess.Move.from_uci(uci_format) in self.legal_moves())

    # check checkmate?
    def checkmate_check(self):
        return self.board.is_checkmate()
    
    # check check?
    def check_check(self):
        return self.board.is_checkmate()
    
    # check stalemate?
    def check_stalemate(self):
        return self.board.is_stalemate()
    
    # check enough material?
    def check_insuffisant_material(self):
        return self.board.is_insufficient_material()
    
    # check game_over?
    def check_gameover(self):
        return self.board.is_game_over()
    
    # check once fiveold repetition or 75 moves without pawn push or capture?
    def check_repetitions(self):
        return (self.board.is_fivefold_repetition(), self.board.is_seventyfive_moves())


    #### CLAIMINGS ####
    def claim_repetitions(self):
        if self.board.can_claim_threefold_repetition():
            return self.board.can_claim_threefold_repetition()
        
        if self.board.can_claim_fifty_moves():
            return self.board.can_claim_fifty_moves()
    
    def claim_draw(self):
        return self.board.can_claim_draw()
    
    
    #### HELPERS ####

    def convert_move_2_string(self, move):

        source_col = Square.get_algeb_not(move.source.col)
        source_row = str(move.source.row)
        target_col = Square.get_algeb_not(move.target.col)
        target_row = str(move.target.row)

        str_source = "".join((source_col,source_row))
        str_target = "".join((target_col,target_row))

        print(str_source, str_target)