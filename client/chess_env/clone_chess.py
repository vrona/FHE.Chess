import chess
from square import Square
from base import *
#-----python-chess-----#
"""
TO SPEED UP DEVELOPMENT AND FACILITATE AI DEVELOPMENT, USE OF PYTHON-CHESS
https://python-chess.readthedocs.io/en/latest/#
"""
class Clone_Chess:
    
    def __init__(self):
        self.board = chess.Board()

    #       __ ___    __        __ 
    #  /\  /    |  | /  \ |\ | (_  
    # /--\ \__  |  | \__/ | \| __) 
                             
    
    # make a move from source to target square
    def move_clone_board(self, move):
       #uci_format = "".join((source,target))
       # checking the moves
       #if self.check_legal_move(move):
        uci_format = self.convert_move_2_string(move)
        self.board.push_san(uci_format)

    def undo_move(self):
            self.board.pop()

    # clearing the board
    def clear_board(self):
        self.board.clear_board()


    #### GETS ####
    # get a snapshot of board
    def get_board(self):
        return self.board
    
    # get the legal moves for a given position
    def legal_moves(self):
        return self.board.legal_moves

    # get the pseudo legal moves (might leaves or put the King in check)) for a given position
    def pseudo_legal_moves(self):
        return self.board.pseudo_legal_moves

    # get the number of legal moves
    def legal_moves_count(self):
        return self.board.legal_moves.count()

    # get game outcome
    def outcome(self, board):
        return board.outcome()


    #  __       __  __             __   __ 
    # /   |__| |_  /   |_/ | |\ | / _  (_  
    # \__ |  | |__ \__ | \ | | \| \__) __)

    # check if move within legal moves
    def check_legal_move(self, move):
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.legal_moves()

    def check_pseudo_legal_move(self, move):
        uci_format = self.convert_move_2_string(move)
        return chess.Move.from_uci(uci_format) in self.pseudo_legal_moves()

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


    #  __                         __   __ 
    # /   |    /\  | |\/| | |\ | / _  (_  
    # \__ |__ /--\ | |  | | | \| \__) __) 

    def claim_repetitions(self):
        if self.board.can_claim_threefold_repetition():
            return self.board.can_claim_threefold_repetition()
        
        if self.board.can_claim_fifty_moves():
            return self.board.can_claim_fifty_moves()
    
    def claim_draw(self):
        return self.board.can_claim_draw()
    

    
    #       __      __   __  __   __ 
    # |__| |_  |   |__) |_  |__) (_  
    # |  | |__ |__ |    |__ | \  __)              

    # convert square[col][row] to string for uci format
    def convert_move_2_string(self, move):

        source_col = Square.get_algeb_not(move.source.col)
        source_row = str(8-move.source.row)
        target_col = Square.get_algeb_not(move.target.col)
        target_row = str(8-move.target.row)

        str_move = "".join((source_col,source_row,target_col,target_row))

        return str_move
    
    # for testing
    def convert_move_2_string_bis(self, source_col, source_row, target_col, target_row):

        src_col = Square.get_algeb_not(source_col)
        src_row = str(8-source_row)
        trgt_col = Square.get_algeb_not(target_col)
        trgt_row = str(8-target_row)

        # str_source = "".join((source_col,source_row))
        # str_target = "".join((target_col,target_row))

        str_move = "".join((src_col,src_row,trgt_col,trgt_row))
        print(str_move)
        return str_move #str_source, str_target

    # for testing
    def convert_string_2_move(self, str_move):
        
        source_target = [x for x in (chess.Move.uci(str_move))]
        source_col,source_row,target_col,target_row = source_target[0], source_target[1], source_target[2], source_target[3]

        source_col = Square.convert_algeb_not(source_col)
        target_col = Square.convert_algeb_not(target_col)

         # micro location
        source = Square(source_row, source_col) 
        target = Square(target_row, target_col)
        
        # move at micro
        return source, target